"""Atomic Redis delivery transitions, fenced by pending-entry ownership."""

from __future__ import annotations

# Check ownership before any write. Repeating a transition after losing its
# response is harmless: the old entry no longer belongs to this consumer.
TRANSITION = """
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 or pending[1][2] ~= ARGV[3] then return 0 end
if ARGV[4] ~= '' then
    if tonumber(ARGV[5]) > 0 then
        redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[5], '*', 'data', ARGV[4])
    else
        redis.call('XADD', KEYS[2], '*', 'data', ARGV[4])
    end
end
redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
redis.call('XDEL', KEYS[1], ARGV[2])
return 1
"""

# A slow, live worker renews only entries it still owns. XCLAIM without this
# check could steal an entry back from its replacement after a disconnection.
RENEW = """
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 or pending[1][2] ~= ARGV[3] then return 0 end
redis.call('XCLAIM', KEYS[1], ARGV[1], ARGV[3], 0, ARGV[2], 'JUSTID')
return 1
"""
