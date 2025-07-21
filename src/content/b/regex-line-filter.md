---
title: Focus your Obsidian note taking - Regex Line Filter
publishDate: 2025-07-21
tags: ["Tech"]
---
I created this plugin for Obsidian out of necessity. I’m a pretty messy note-taker, especially when I’m forced to focus more on getting the idea/task/reminder down _ASAP_ rather than sparing a thought for how a note should be structured. In a recent project this meant primarily a single note, divided by headings and a date, everything else was mostly dot-point lists and indentation. This led to long notes and an eventual traffic-lights system for todo, doing, done Eg.

```
### @today
- thing A
  - sub-thing 1 🟢
  - sub-thing n 🟢
- thing B 🟠
  - sub thing 1 🟢
  - sub thing 2 - sub thing n 🔴
```

Eventually I needed a quick way to check which tasks still needed action/completion so I thought that being able to filter for all lines with certain regex matches could be useful way to get a quick view of where things are at. One thing led to another and then to the regex-line-filter plugin.

Here is demo of what it looked like at v1.0  
![regex-line-filter demo](https://64mm4-kn1f3.github.io/5RV/regex-line-filter.gif)

Since then some community members suggested some great features in github which are now all implemented:

- Recently used filters can be pinned or saved.
- Saved ‘custom’ filters can be named and hotkey-assigned.
- Custom filters can be toggled on in tandem, widening or narrowing the lines displayed in the filtered view
- Child indents of a matched line can be included in filtered view
- Date-based template variable ‘literals’ can be injected into filters so you can quickly filter for {{today}}, {{yesterday}}, {{last-month}} or custom date formats using {{date:YYYY-MM-DD}} variations.

Hope some folks in the Obsidian community find it useful! Search for `regex-line-filter` in Community Plugins if you’re curious.

Repo here: [github.com/64MM4-KN1F3/regex-line-filter](https://github.com/64MM4-KN1F3/regex-line-filter)
