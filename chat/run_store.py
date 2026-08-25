from backend.core.conf import settings
from backend.database.redis import RedisCli, redis_client

_ACTIVE_LEASE_SECONDS = 30

_CLAIM_SCRIPT = """
if not redis.call('set', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) then
    return 0
end
redis.call('del', KEYS[2])
return 1
"""

_POLL_SCRIPT = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
if redis.call('exists', KEYS[2]) == 1 then
    return 1
end
if ARGV[2] == '1' then
    redis.call('expire', KEYS[1], ARGV[3])
end
return 2
"""

_REQUEST_CANCEL_SCRIPT = """
if redis.call('exists', KEYS[1]) == 0 then
    return 0
end
redis.call('set', KEYS[2], '1', 'EX', ARGV[1])
return 1
"""

_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('del', KEYS[1], KEYS[2])
return 1
"""


class SharedChatRunStore:
    """基于 Redis 的跨进程聊天任务协调存储"""

    def __init__(self, client: RedisCli = redis_client) -> None:
        self.client = client
        self.key_prefix = f'{settings.PLUGIN_REDIS_PREFIX}:ai:chat-run'

    def _key(self, conversation_id: str, suffix: str) -> str:
        return f'{self.key_prefix}:{conversation_id}:{suffix}'

    async def claim(self, *, conversation_id: str, run_id: str) -> bool:
        """
        获取任务执行租约

        :param conversation_id: 对话 ID
        :param run_id: 运行 ID
        :return:
        """
        result = await self.client.eval(
            _CLAIM_SCRIPT,
            2,
            self._key(conversation_id, 'owner'),
            self._key(conversation_id, 'cancel'),
            run_id,
            _ACTIVE_LEASE_SECONDS,
        )
        return bool(result)

    async def is_active(self, *, conversation_id: str) -> bool:
        """
        判断任务租约是否有效

        :param conversation_id: 对话 ID
        :return:
        """
        return bool(await self.client.exists(self._key(conversation_id, 'owner')))

    async def poll(self, *, conversation_id: str, run_id: str, renew: bool) -> int:
        """
        检查租约和取消信号，并按需续期

        :param conversation_id: 对话 ID
        :param run_id: 运行 ID
        :param renew: 是否续期租约
        :return: 0 表示租约丢失，1 表示请求取消，2 表示继续运行
        """
        return int(
            await self.client.eval(
                _POLL_SCRIPT,
                2,
                self._key(conversation_id, 'owner'),
                self._key(conversation_id, 'cancel'),
                run_id,
                '1' if renew else '0',
                _ACTIVE_LEASE_SECONDS,
            )
        )

    async def request_cancel(self, *, conversation_id: str) -> bool:
        """
        设置跨进程取消信号

        :param conversation_id: 对话 ID
        :return:
        """
        result = await self.client.eval(
            _REQUEST_CANCEL_SCRIPT,
            2,
            self._key(conversation_id, 'owner'),
            self._key(conversation_id, 'cancel'),
            _ACTIVE_LEASE_SECONDS,
        )
        return bool(result)

    async def release(self, *, conversation_id: str, run_id: str) -> bool:
        """
        释放当前任务租约和取消信号

        :param conversation_id: 对话 ID
        :param run_id: 运行 ID
        :return:
        """
        result = await self.client.eval(
            _RELEASE_SCRIPT,
            2,
            self._key(conversation_id, 'owner'),
            self._key(conversation_id, 'cancel'),
            run_id,
        )
        return bool(result)


shared_chat_run_store = SharedChatRunStore()
