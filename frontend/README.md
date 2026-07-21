# Dashboard

An ops view over a recorded run of the pipeline in `../src/rdi`.

There is no backend. `rdi-export` writes `public/snapshot.json` from an actual pipeline run and
the page renders that, so the whole thing builds static and deploys anywhere. The live feed
replays the recorded events at a readable pace: the pacing is synthetic, the events, the
predictions, the log lines and every number are not.

## Run it

```bash
cd ..
make dashboard-data     # writes frontend/public/snapshot.json
make dashboard          # next dev on :3000
```

Or from here:

```bash
npm install
npm run dev
```

## Regenerating the data

`make dashboard-data` runs the real pipeline, so the snapshot changes when the pipeline does.
It ships committed, which means `npm run build` works on a clean clone without Python.

## Layout

```
app/page.tsx            server component, reads the snapshot and lays out the panels
components/LiveFeed.tsx the replaying event feed and the inspector
components/primitives   Panel, Stat, Bar
lib/types.ts            the snapshot shape, plus one colour per incident class
```

Colour carries meaning here rather than decoration: each incident class keeps the same hue
everywhere, rose marks the class the metrics get wrong, and emerald marks a number that beat
its baseline.
