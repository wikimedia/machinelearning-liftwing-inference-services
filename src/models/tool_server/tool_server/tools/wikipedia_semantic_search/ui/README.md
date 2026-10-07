# grid.html

The card grid View for the MCP App version of `wikipedia_semantic_search`
(`../cards.py`). It is a build artifact: a single-file Vite bundle (Vue +
Codex + the MCP Apps SDK), checked in so the image needs no Node build.

## Source

- Repo: https://gitlab.wikimedia.org/egardner/wikipedia-semantic-search-mcp
- Commit: `bd9e37257e3a9597733d7f51ba28e1477661d551` (2026-05-03)
- Plus one local change, not yet upstream (below)

## Local change

`ui/useMcpHost.ts` holds the SDK `App` instance in a `shallowRef` instead
of a `ref`. A deep `ref` wraps the instance in a reactive proxy, and Vue
then reads `__v_isRef` on the cross-origin `window.parent` the SDK holds.
In hosts that give the View an opaque origin, such as LibreChat, that throws
`SecurityError`, so clicking a card opens nothing.

```diff
-import { ref, type Ref } from 'vue';
+import { ref, shallowRef, type Ref } from 'vue';
...
-	const app = ref<App | null>( null );
+	const app = shallowRef<App | null>( null );
```

## Rebuild

```sh
git clone https://gitlab.wikimedia.org/egardner/wikipedia-semantic-search-mcp
cd wikipedia-semantic-search-mcp
git checkout bd9e37257e3a9597733d7f51ba28e1477661d551
# apply the change above
npm ci
npm run build:ui
cp dist/index.html <this directory>/grid.html
```

The payload the View expects is `SearchPayload` in `ui/types.ts` there;
`cards.to_payload` builds it. Image hosts the View may load from are
`cards.RESOURCE_DOMAINS` (the View's CSP).
