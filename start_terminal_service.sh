#!/bin/bash
python parlai/chat_service/services/terminal_chat/run.py    --config-path  parlai/chat_service/tasks/covid19_chatbot/config.yml >/dev/null 2>&1 &
sleep 6
python parlai/chat_service/services/terminal_chat/client.py
