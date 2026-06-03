#!/bin/sh
/usr/sbin/apache2 -d /srv/app -f /srv/app/apache2.conf -DFOREGROUND -k start
