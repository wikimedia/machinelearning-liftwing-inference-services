import logging
from abc import ABC, abstractmethod
from typing import Any

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster, ExecutionProfile
from cassandra.cqlengine import connection
from cassandra.cqlengine.models import Model
from cassandra.policies import DCAwareRoundRobinPolicy
from pydantic_settings import BaseSettings


class CassandraSettings(BaseSettings):
    """
    Cassandra cache configuration settings.

    Reads configuration from environment variables with type validation and defaults.
    All settings can be overridden via environment variables.

    Attributes:
        cassandra_servers: Semicolon-separated list of Cassandra host addresses
        cassandra_keyspace: Keyspace name to use
        cassandra_table: Table name to use for caching
        cassandra_user: Username for authentication
        cassandra_password: Password for authentication
        cassandra_datacenter: Datacenter name for load balancing
        cassandra_ttl: Default time-to-live for cache entries in seconds
        cassandra_request_timeout: Timeout for Cassandra queries in seconds
        cassandra_protocol_version: Cassandra native protocol version to use
    """

    cassandra_servers: str = "127.0.0.1"
    cassandra_keyspace: str = "keyspace"
    cassandra_table: str = "table"
    cassandra_user: str = "cassandra"
    cassandra_password: str = "cassandra"
    cassandra_datacenter: str = "datacenter1"
    cassandra_ttl: int = 3600
    cassandra_request_timeout: int = 1
    cassandra_protocol_version: int = 4


class BaseCassandraCache(ABC):
    """
    Abstract base class for Cassandra-backed caching implementations.

    This class provides the foundation for creating cache implementations for
    ML model predictions. It handles all Cassandra connection and configuration
    details, allowing subclasses to focus on their specific caching logic.

    Subclasses must implement:
        - from_cache(): Logic to retrieve cached data
        - to_cache(): Logic to store data in cache

    The base class automatically:
        - Sets up Cassandra connection with authentication
        - Configures keyspace and TTL settings
        - Syncs table schema on initialization

    Connection management is handled per service instance, with configuration
    read from environment variables.
    """

    def __init__(
        self,
        cache_model_class: type[Model],
        settings: CassandraSettings | None = None,
    ) -> None:
        """
        Initialize Cassandra cache connection and sync table schema.

        Sets up connection to Cassandra cluster, configures the keyspace and TTL
        for the provided model class, and synchronizes the table schema. All
        configuration is read from environment variables via Pydantic settings.

        Args:
            cache_model_class: CQL model class (subclass of Model) that defines
                the Cassandra table schema
            settings: CassandraSettings instance. If None, will be created from
                environment variables.

        Raises:
            Exception: If Cassandra connection setup or table sync fails. The
                original exception is logged and re-raised.
            ValidationError: If required environment variables are missing or invalid.

        Environment Variables:
            CASSANDRA_SERVERS: Semicolon-separated host list (default: 127.0.0.1)
            CASSANDRA_KEYSPACE: Keyspace name (default: keyspace)
            CASSANDRA_TABLE: Table name to use for caching (default: table)
            CASSANDRA_USER: Authentication username (default: cassandra)
            CASSANDRA_PASSWORD: Authentication password (default: cassandra)
            CASSANDRA_DATACENTER: Datacenter name for load balancing (default: datacenter1)
            CASSANDRA_TTL: Cache entry TTL in seconds (default: 3600)
            CASSANDRA_REQUEST_TIMEOUT: Query timeout in seconds (default: 1)
            CASSANDRA_PROTOCOL_VERSION: Protocol version (default: 4)
        """
        if settings is None:
            settings = CassandraSettings()

        self.settings = settings
        self.ttl = settings.cassandra_ttl

        cache_model_class.__keyspace__ = settings.cassandra_keyspace
        cache_model_class.__table_name__ = settings.cassandra_table
        cache_model_class.__options__ = {"default_time_to_live": self.ttl}

        serverlist = settings.cassandra_servers.split(";")

        exc_profiles = {}
        auth_provider = PlainTextAuthProvider(
            username=settings.cassandra_user,
            password=settings.cassandra_password,
        )

        for s in serverlist:
            exc_profiles[s] = ExecutionProfile(
                load_balancing_policy=DCAwareRoundRobinPolicy(
                    local_dc=settings.cassandra_datacenter
                ),
                request_timeout=settings.cassandra_request_timeout,
            )

        logging.info(
            f"Setting up Cassandra connection. Servers: {serverlist} "
            f"Keyspace: {settings.cassandra_keyspace} Table: {settings.cassandra_table} "
            f"Datacenter: {settings.cassandra_datacenter} Row TTL: {self.ttl}"
        )

        try:
            connection.setup(
                hosts=serverlist,
                default_keyspace=settings.cassandra_keyspace,
                retry_connect=True,
                lazy_connect=False,
                protocol_version=settings.cassandra_protocol_version,
                execution_profiles=exc_profiles,
                auth_provider=auth_provider,
            )

            # Validate that the table exists
            self._validate_table_exists(
                serverlist=serverlist,
                auth_provider=auth_provider,
                keyspace=settings.cassandra_keyspace,
                table=settings.cassandra_table,
            )

            logging.info(
                f"Cassandra connection successful. Validated table exists: "
                f"{settings.cassandra_keyspace}.{settings.cassandra_table}"
            )
        except Exception as e:
            logging.error(f"Failed to setup Cassandra connection: {e}")
            raise

    def _validate_table_exists(
        self,
        serverlist: list[str],
        auth_provider: PlainTextAuthProvider,
        keyspace: str,
        table: str,
    ) -> None:
        """
        Validate that the specified table exists in Cassandra.

        Args:
            serverlist: List of Cassandra host addresses
            auth_provider: Authentication provider
            keyspace: Keyspace name
            table: Table name

        Raises:
            ValueError: If the table does not exist
        """
        cluster = Cluster(serverlist, auth_provider=auth_provider)
        try:
            session = cluster.connect(keyspace)
            # Query system tables to check if table exists
            query = """
                SELECT table_name FROM system_schema.tables
                WHERE keyspace_name = %s AND table_name = %s
            """
            result = session.execute(query, (keyspace, table))
            if not result.one():
                raise ValueError(
                    f"Table '{keyspace}.{table}' does not exist. "
                    f"Please create the table schema before starting the service."
                )
        finally:
            cluster.shutdown()

    @abstractmethod
    def from_cache(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Retrieve data from cache. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def to_cache(self, *args: Any, **kwargs: Any) -> None:
        """Store data in cache. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def remove_from_cache(self, *args: Any, **kwargs: Any) -> None:
        """Remove data from cache. Must be implemented by subclasses."""
        pass
