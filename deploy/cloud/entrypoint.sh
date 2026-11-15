#!/bin/sh
# Container entrypoint for deploy/cloud: load API keys from a mounted secrets
# file into this process's environment, then run the TradingAgents CLI.
#
# The file is mounted as a Docker secret (/run/secrets/tradingagents.env), so
# keys are never baked into the image, passed on a command line, or shown by
# `docker inspect`. Nothing here prints a key or a value. Platforms that inject
# secrets as environment variables directly need no file; it is then skipped.
#
# Format: one NAME=value per line; blank lines and # comments are ignored; a
# value may be wrapped in double quotes. The value is never evaluated by a shell.
set -eu

secrets_file="${TRADINGAGENTS_SECRETS_FILE:-/run/secrets/tradingagents.env}"

if [ -f "$secrets_file" ]; then
    if [ ! -r "$secrets_file" ]; then
        echo "entrypoint: $secrets_file is not readable by this user (uid $(id -u)); see deploy/cloud/README.md" >&2
        exit 1
    fi
    cr=$(printf '\r')
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%"$cr"}
        case "$line" in
            '' | '#'*) continue ;;
        esac
        name=${line%%=*}
        if [ "$name" = "$line" ]; then
            echo "entrypoint: skipped a secrets line with no '='" >&2
            continue
        fi
        case "$name" in
            '' | [0-9]* | *[!A-Za-z0-9_]*)
                echo "entrypoint: skipped a secrets line with an invalid name" >&2
                continue ;;
        esac
        value=${line#*=}
        case "$value" in
            \"*\") value=${value#\"}; value=${value%\"} ;;
        esac
        [ -n "$value" ] || continue
        export "$name=$value"
    done < "$secrets_file"
fi

exec tradingagents "$@"
