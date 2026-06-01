# Deploy — worldcup.polyalpha.cn

Mirrors the existing polyalpha.cn pattern (static site served locally + cloudflared tunnel +
launchd). Nothing here is active until you bootstrap it. Fill the `<...>` placeholders first.

## 1. Serve the static dashboard
`site/` is fully static (`index.html` + generated `data.json`). Serve it on a local port:

```bash
python3 -m http.server 8780 --directory /Users/bot/worldcup/site
```

(or copy `site/` into your existing polyalpha frontend deploy and push to Cloudflare Pages).

## 2. Expose via cloudflared tunnel
Add the ingress rule from `cloudflared-worldcup.yml.example` to your tunnel config and add a
DNS CNAME for `worldcup.polyalpha.cn` pointing at the tunnel. With the existing tunnel:

```bash
cloudflared tunnel route dns <TUNNEL_NAME> worldcup.polyalpha.cn
# then add the ingress hostname->service rule and restart the tunnel
```

## 3. Daily refresh (during the tournament)
`review` pulls fresh scores, scores completed matches vs our prior predictions, re-predicts,
re-simulates, and republishes `data.json`. Schedule it with launchd:

```bash
cp deploy/com.worldcup.review.plist.example ~/Library/LaunchAgents/com.worldcup.review.plist
# edit paths inside, then:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.worldcup.review.plist
```

Two daily runs are sensible during the group stage (morning EU + before US kickoffs). Adjust the
`StartCalendarInterval` blocks. Local-routine cron UI can mislabel schedules — trust the plist,
verify with `launchctl print gui/$(id -u)/com.worldcup.review`.
