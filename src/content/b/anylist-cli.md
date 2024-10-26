---
title: A Simple AnyList CLI
publishDate: 2024-10-24
tags: ["Tech"]
---

If you cohabitate and have shared-grocery needs, a shared list is useful.

I quite like AnyList's mobile app but because I spend so much time in front of the laptop (and type way faster on that) I wanted a simple way to add items to our shared list. I couldn't find anything online so put a simple CLI together using Node.JS making use of [codetheweb's AnyList API tools](https://github.com/codetheweb/anylist)

If you'd like to try it out, all you need is git and Node installed.

Create a directory for this tool to live in then:
```shell
git clone https://github.com/64MM4-KN1F3/anylist_cli.git
```

Now create a file in the same path called `.env` with the following lines:
```sh
EMAIL=<your_anylist_username>
PASSWORD=<your_anylist_password>
PRIMARY_LIST_NAME=<the_list_you_want_to_manage_via_CLI>
```

Install dependencies: 
```shell
npm install
```

Run:
```shell
node anyList_CLI.js
```

Away you go. Start adding items!

To quit, type: 'q', 'quit' or 'exit'
