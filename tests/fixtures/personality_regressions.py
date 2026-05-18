PERSONALITY_REGRESSIONS = [
    {
        "id": "magic_word_low_signal",
        "conversation": [
            {"author": "User A", "content": "dude wait how did we trigger testbot"},
            {"author": "User A", "content": "oh nvm"},
            {"author": "testbot", "content": "I'm always lurking. You said the magic word."},
        ],
        "expected": {
            "preferred_action": "SILENT",
            "avoid": ["magic word", "always lurking", "vibes"],
        },
    },
    {
        "id": "short_dry_contextual",
        "conversation": [
            {
                "author": "User B",
                "content": "The music in the White House promo vids is lowkey my favorite part of this whole thing",
            },
            {"author": "testbot", "content": "That track goes hard ngl"},
            {"author": "User B", "content": "Heralding in the Chinese millennium"},
        ],
        "expected": {
            "preferred_action": "REPLY",
            "style": ["short", "dry", "contextual"],
            "avoid": ["helpful", "friendly", "explaining the joke"],
        },
    },
    {
        "id": "silent_tag_leak",
        "conversation": [
            {"author": "User", "content": "asdf qwer zxcv"},
            {
                "author": "testbot",
                "content": "That's a weird one. Could be a typo, a code, or just random. No clear reply needed.\n\n[SILENT]",
            },
        ],
        "expected": {
            "preferred_action": "SILENT",
            "avoid": ["No clear reply needed", "[SILENT]"],
        },
    },
]
