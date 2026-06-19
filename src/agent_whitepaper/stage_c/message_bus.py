"""阶段C-2：异步消息总线 - Agent间通信的基础设施。

第一性原理：
    多Agent协作需要通信。消息总线提供：
    - 点对点消息：Agent A → Agent B
    - 广播消息：Agent A → 所有Agent
    - 主题订阅：Agent订阅感兴趣的主题

    设计采用同步队列实现（简化版，避免asyncio复杂性）：
    每个Agent有一个消息队列，发送消息就是往队列里放，
    接收消息就是从队列里取。

    这实现了"操作多个agent"的基础：agent间可以互相通信协调。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from collections import deque
import time
import uuid


# ============================================================
# 消息类型
# ============================================================
class MessageType(Enum):
    """消息类型。"""
    REQUEST = "request"      # 请求
    RESPONSE = "response"    # 响应
    BROADCAST = "broadcast"  # 广播
    NOTIFICATION = "notification"  # 通知
    COORDINATION = "coordination"  # 协调


# ============================================================
# 消息
# ============================================================
@dataclass(frozen=True)
class Message:
    """一条消息。

    Attributes:
        message_id: 消息唯一ID
        sender_id: 发送者Agent ID
        receiver_id: 接收者Agent ID（None表示广播）
        message_type: 消息类型
        content: 消息内容
        timestamp: 发送时间戳
        reply_to: 回复的消息ID（如果是响应）
        priority: 优先级（0最高）
    """
    message_id: str
    sender_id: str
    receiver_id: Optional[str]
    message_type: MessageType
    content: Any
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    priority: int = 0


# ============================================================
# 消息队列（带优先级）
# ============================================================
class MessageQueue:
    """带优先级的消息队列。"""

    def __init__(self, max_size: int = 1000):
        if max_size < 1:
            raise ValueError("max_size 必须 >= 1")
        self.max_size = max_size
        self._queue: deque = deque()

    def push(self, message: Message) -> bool:
        """推入消息。"""
        if len(self._queue) >= self.max_size:
            return False
        # 按优先级插入（简化：priority小的在前）
        # 对于小队列，简单追加即可
        self._queue.append(message)
        return True

    def pop(self) -> Optional[Message]:
        """弹出最高优先级的消息。"""
        if not self._queue:
            return None
        # 找到优先级最高（priority值最小）的消息
        # 对于小队列，线性扫描足够
        if len(self._queue) == 1:
            return self._queue.popleft()
        # 找最小priority的索引
        min_idx = 0
        min_pri = self._queue[0].priority
        for i in range(1, len(self._queue)):
            if self._queue[i].priority < min_pri:
                min_pri = self._queue[i].priority
                min_idx = i
        # 移除并返回
        msg = self._queue[min_idx]
        del self._queue[min_idx]
        return msg

    def peek(self) -> Optional[Message]:
        """查看但不弹出。"""
        if not self._queue:
            return None
        return self._queue[0]

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()


# ============================================================
# 消息总线
# ============================================================
class MessageBus:
    """消息总线：管理所有Agent间的消息路由。

    功能：
    1. 注册/注销Agent的消息队列
    2. 点对点消息发送
    3. 广播消息
    4. 主题订阅与发布
    5. 消息统计
    """

    def __init__(self):
        self._queues: Dict[str, MessageQueue] = {}
        self._subscriptions: Dict[str, List[str]] = {}  # topic -> [agent_ids]
        self._stats: Dict[str, int] = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_dropped": 0,
            "broadcasts": 0,
        }

    def register_agent(self, agent_id: str, max_queue_size: int = 1000) -> bool:
        """注册Agent的消息队列。"""
        if agent_id in self._queues:
            return False
        self._queues[agent_id] = MessageQueue(max_queue_size)
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """注销Agent。"""
        if agent_id not in self._queues:
            return False
        del self._queues[agent_id]
        # 从所有订阅中移除
        for topic, subscribers in self._subscriptions.items():
            if agent_id in subscribers:
                subscribers.remove(agent_id)
        return True

    def send(self, message: Message) -> bool:
        """发送点对点消息。"""
        self._stats["total_sent"] += 1
        if message.receiver_id is None:
            return self.broadcast(message)

        if message.receiver_id not in self._queues:
            self._stats["total_dropped"] += 1
            return False

        success = self._queues[message.receiver_id].push(message)
        if success:
            self._stats["total_delivered"] += 1
        else:
            self._stats["total_dropped"] += 1
        return success

    def broadcast(self, message: Message) -> bool:
        """广播消息给所有Agent（除了发送者）。"""
        self._stats["broadcasts"] += 1
        self._stats["total_sent"] += 1
        delivered = 0
        for agent_id, queue in self._queues.items():
            if agent_id == message.sender_id:
                continue  # 不发给自己
            if queue.push(message):
                delivered += 1
        self._stats["total_delivered"] += delivered
        return delivered > 0

    def publish(self, topic: str, message: Message) -> bool:
        """发布主题消息给订阅者。"""
        self._stats["total_sent"] += 1
        subscribers = self._subscriptions.get(topic, [])
        delivered = 0
        for agent_id in subscribers:
            if agent_id == message.sender_id:
                continue
            if agent_id in self._queues:
                if self._queues[agent_id].push(message):
                    delivered += 1
        self._stats["total_delivered"] += delivered
        return delivered > 0

    def subscribe(self, agent_id: str, topic: str) -> bool:
        """Agent订阅主题。"""
        if agent_id not in self._queues:
            return False
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        if agent_id not in self._subscriptions[topic]:
            self._subscriptions[topic].append(agent_id)
        return True

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Agent取消订阅。"""
        if topic not in self._subscriptions:
            return False
        if agent_id in self._subscriptions[topic]:
            self._subscriptions[topic].remove(agent_id)
            return True
        return False

    def receive(self, agent_id: str) -> Optional[Message]:
        """Agent接收一条消息。"""
        if agent_id not in self._queues:
            return None
        return self._queues[agent_id].pop()

    def receive_all(self, agent_id: str) -> List[Message]:
        """Agent接收所有待处理消息。"""
        if agent_id not in self._queues:
            return []
        messages = []
        while True:
            msg = self._queues[agent_id].pop()
            if msg is None:
                break
            messages.append(msg)
        return messages

    def has_messages(self, agent_id: str) -> bool:
        """检查Agent是否有待处理消息。"""
        if agent_id not in self._queues:
            return False
        return len(self._queues[agent_id]) > 0

    def get_stats(self) -> dict:
        """获取消息总线统计。"""
        return {
            **self._stats,
            "registered_agents": len(self._queues),
            "total_topics": len(self._subscriptions),
            "queue_sizes": {
                aid: len(q) for aid, q in self._queues.items()
            },
        }

    def create_message(
        self,
        sender_id: str,
        receiver_id: Optional[str],
        message_type: MessageType,
        content: Any,
        priority: int = 0,
        reply_to: Optional[str] = None
    ) -> Message:
        """创建消息的便捷方法。"""
        return Message(
            message_id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            reply_to=reply_to,
        )
