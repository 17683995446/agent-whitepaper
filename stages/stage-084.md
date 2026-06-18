# 阶段84：禁止修改系统关键文件的规则引擎

> **阶段编号**：S84 | **所属层**：第九层·安全护栏层
>
> **核心概念通俗映射**：
> | 本阶段概念 | 通俗理解 |
> |-----------|---------|
> | 受保护文件列表 | "禁区地图"——哪些文件绝对不能碰 |
> | 文件操作拦截 | "门口安检"——每次文件操作都要检查是否允许 |
> | 规则匹配 | "查名单"——用正则匹配文件路径是否在保护名单上 |
> | 违规日志 | "违章记录"——谁、什么时候、想改什么文件 |
> | FileProtectionEngine | "文件保护的完整画像"——组合StorageCleaner，在存储清理基础上增加文件保护状态 |

---

## 1. 第一性原理依据

本阶段从阶段83（存储清理）的核心结论出发，建立禁止修改系统关键文件的规则引擎。核心依据为：

- **A1.8（生存条件）**（引用阶段1）：智能体的生存条件包括"不破坏宿主系统"。修改系统关键文件（如/boot、/etc/passwd）可能导致宿主系统崩溃或安全漏洞，直接违反A1.8的生存条件。本阶段将"不破坏宿主系统"这一生存条件工程化为可执行的文件保护规则。

- **A83.1-A83.5（存储清理）**（引用阶段83）：存储清理涉及文件删除操作。文件保护规则引擎是存储清理的"安全带"——确保清理操作不会误删系统关键文件。两者协同工作：存储清理负责释放空间，文件保护负责防止误操作。

本阶段的核心目标是：**定义受保护文件列表 [/boot, /etc/passwd, /etc/shadow, C:\Windows\System32, 自身启动代码]（A84.1），文件操作拦截 intercept(file_op) → allow/deny（A84.2），规则匹配 algorithm = regex_match(file_path, protected_patterns)（A84.3），违规日志 log_violation(operation, file_path, timestamp)（A84.4），封装为FileProtectionEngine数据类，组合StorageCleaner作为存储清理的输入。**

> **去重声明**：本阶段仅定义文件保护的规则引擎，不重新定义存储清理（阶段83）或生存条件（阶段1）。文件保护是存储清理的安全约束层，确保清理操作不误删关键文件。

---

## 2. 核心数学公式

### 第一部分：受保护文件列表

#### 2.1 受保护文件列表定义（A84.1）

受保护文件列表定义了智能体绝对不能修改的文件和目录：

$$\mathcal{P} = \{p_1, p_2, \ldots, p_K\} \tag{A84.1}$$

**Linux系统受保护路径**：

| 路径模式 | 保护级别 | 说明 |
|---------|---------|------|
| /boot/** | 绝对保护 | 引导加载器，修改后系统无法启动 |
| /etc/passwd | 绝对保护 | 用户账户信息 |
| /etc/shadow | 绝对保护 | 用户密码哈希 |
| /etc/sudoers | 绝对保护 | sudo权限配置 |
| /etc/ssh/** | 绝对保护 | SSH配置和密钥 |
| /etc/systemd/** | 绝对保护 | 系统服务配置 |
| /proc/** | 绝对保护 | 内核虚拟文件系统 |
| /sys/** | 绝对保护 | 内核设备树 |
| /dev/** | 绝对保护 | 设备文件 |
| /usr/bin/** | 条件保护 | 系统二进制文件 |
| /usr/lib/** | 条件保护 | 系统库文件 |

**Windows系统受保护路径**：

| 路径模式 | 保护级别 | 说明 |
|---------|---------|------|
| C:\Windows\System32\** | 绝对保护 | 系统核心文件 |
| C:\Windows\SysWOW64\** | 绝对保护 | 64位系统文件 |
| C:\Windows\Win.ini | 绝对保护 | Windows初始化文件 |
| C:\ProgramData\** | 条件保护 | 程序共享数据 |
| HKEY_LOCAL_MACHINE\** | 绝对保护 | 注册表核心键 |

**自身启动代码保护**：

智能体自身的启动代码也必须保护，防止自我修改导致功能异常：

$$\mathcal{P}_{\text{self}} = \{\text{agent\_executable}, \text{agent\_config}, \text{agent\_startup\_script}\} \tag{A84.1-self}$$

| 自身路径 | 保护级别 | 说明 |
|---------|---------|------|
| agent主程序 | 绝对保护 | 修改后可能行为异常 |
| agent配置文件 | 绝对保护 | 修改后可能绕过安全约束 |
| agent启动脚本 | 绝对保护 | 修改后可能以不同权限启动 |
| agent数据目录 | 条件保护 | 可在安全约束内修改 |

**保护级别的区分**：

$$\text{protection}(p) = \begin{cases} \text{ABSOLUTE} & p \in \mathcal{P}_{\text{absolute}} \\ \text{CONDITIONAL} & p \in \mathcal{P}_{\text{conditional}} \end{cases} \tag{A84.1-level}$$

- **绝对保护**：任何情况下都不允许修改、删除或重命名
- **条件保护**：仅在明确授权的情况下允许读取，不允许写入

- **输入**：操作系统类型、智能体安装路径
- **输出**：受保护路径列表 $\mathcal{P}$
- **约束**：列表不可在运行时修改

> **白话翻译**：受保护文件列表就是"禁区地图"——哪些文件绝对不能碰。分两类：绝对保护（碰都不能碰，如/boot、/etc/passwd、自身启动代码）和条件保护（只能看不能改，如/usr/bin）。列表在运行时不能修改，防止智能体自己把保护规则删了。

---

### 第二部分：文件操作拦截

#### 2.2 文件操作拦截定义（A84.2）

所有文件操作在执行前必须经过拦截检查：

$$\text{intercept}(\text{file\_op}) \to \text{allow} \mid \text{deny} \tag{A84.2}$$

**拦截的文件操作类型**：

| 操作类型 | 拦截策略 | 说明 |
|---------|---------|------|
| WRITE | 必须检查 | 写入是最危险的操作 |
| DELETE | 必须检查 | 删除不可逆 |
| RENAME | 必须检查 | 重命名可能绕过保护 |
| CHMOD | 必须检查 | 修改权限可能绕过保护 |
| CREATE | 必须检查 | 创建新文件可能覆盖关键文件 |
| READ | 条件检查 | 仅条件保护路径需检查 |
| EXECUTE | 不检查 | 执行不修改文件 |

**拦截的执行流程**：

```
file_operation(path, op_type)
    │
    ├──→ resolve_real_path(path)     # 解析符号链接
    │
    ├──→ regex_match(real_path, P)   # 规则匹配（A84.3）
    │
    ├──→ if match:
    │       ├──→ ABSOLUTE → deny
    │       └──→ CONDITIONAL + WRITE → deny
    │
    ├──→ if deny:
    │       ├──→ log_violation()     # 违规日志（A84.4）
    │       └──→ return DENY
    │
    └──→ return ALLOW
```

**符号链接解析**：

智能体可能通过符号链接绕过保护规则。拦截器必须解析所有符号链接：

$$\text{real\_path} = \text{realpath}(\text{file\_path}) \tag{A84.2-realpath}$$

解析后的真实路径用于规则匹配，防止通过 `/tmp/link → /etc/passwd` 的方式绕过保护。

**路径规范化**：

除符号链接外，还需处理路径中的 `..`、`.`、`//` 等：

$$\text{normalized\_path} = \text{normpath}(\text{real\_path}) \tag{A84.2-normpath}$$

规范化确保 `/etc/../etc/passwd` 被正确识别为 `/etc/passwd`。

- **输入**：文件操作（路径+操作类型）
- **输出**：allow / deny
- **约束**：拦截延迟 < 1ms

> **白话翻译**：文件操作拦截就是"门口安检"——每次文件操作都要检查是否允许。写入、删除、重命名、改权限、创建新文件都要检查，读取只在条件保护路径检查。特别注意符号链接——不能通过创建链接绕过保护。拦截必须快，延迟不超过1毫秒。

---

### 第三部分：规则匹配

#### 2.3 规则匹配定义（A84.3）

规则匹配使用正则表达式检查文件路径是否匹配受保护模式：

$$\text{match}(\text{file\_path}, \mathcal{P}) = \mathbb{1}\left[\exists p \in \mathcal{P} : \text{regex\_match}(\text{file\_path}, p)\right] \tag{A84.3}$$

**正则表达式的模式设计**：

受保护路径使用通配符模式，转换为正则表达式进行匹配：

| 路径模式 | 正则表达式 | 说明 |
|---------|-----------|------|
| /boot/** | `^/boot/.*` | /boot下所有文件 |
| /etc/passwd | `^/etc/passwd$` | 精确匹配 |
| /etc/ssh/** | `^/etc/ssh/.*` | /etc/ssh下所有文件 |
| C:\Windows\System32\** | `^C:\\Windows\\System32\\.*` | System32下所有文件 |

**通配符到正则的转换规则**：

| 通配符 | 正则 | 说明 |
|--------|------|------|
| ** | .* | 匹配任意深度子目录 |
| * | [^/]* | 匹配当前目录下的任意文件名 |
| ? | . | 匹配单个字符 |

**规则匹配的性能优化**：

当受保护路径数量较多时，逐个正则匹配开销大。采用前缀树（Trie）优化：

$$\text{Trie}(\mathcal{P}) \implies O(|\text{path}|) \text{ lookup} \tag{A84.3-trie}$$

前缀树将匹配复杂度从 $O(K \cdot |\text{path}|)$ 降低到 $O(|\text{path}|)$，其中 $K$ 为受保护路径数量。

**规则匹配的缓存**：

对已检查过的路径缓存结果：

$$\text{cache}(\text{path}) = \begin{cases} \text{cached\_result} & \text{if path in cache} \\ \text{regex\_match}(\text{path}, \mathcal{P}) & \text{otherwise} \end{cases} \tag{A84.3-cache}$$

缓存大小限制为10000条，采用LRU淘汰策略。

- **输入**：文件路径、受保护模式列表
- **输出**：是否匹配、匹配的保护级别
- **约束**：匹配延迟 < 0.5ms

> **白话翻译**：规则匹配就是"查名单"——用正则匹配文件路径是否在保护名单上。/boot/**匹配/boot下所有文件，/etc/passwd精确匹配。路径多了逐个查太慢，用前缀树优化——按路径前缀快速定位。查过的路径缓存结果，下次直接用。

---

### 第四部分：违规日志

#### 2.4 违规日志定义（A84.4）

每次被拦截的文件操作都记录违规日志：

$$\text{log\_violation}(\text{operation}, \text{file\_path}, \text{timestamp}) \tag{A84.4}$$

**违规日志的完整字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | float | 违规时间戳 |
| operation | str | 操作类型（WRITE/DELETE/RENAME/CHMOD/CREATE） |
| file_path | str | 目标文件路径 |
| real_path | str | 解析后的真实路径 |
| matched_pattern | str | 匹配的受保护模式 |
| protection_level | str | 保护级别（ABSOLUTE/CONDITIONAL） |
| caller_info | str | 调用方信息（函数名+行号） |
| action_taken | str | 采取的动作（DENY） |

**违规日志的存储**：

违规日志持久化存储，防止智能体修改或删除日志：

$$\text{violation\_log} \to \text{append\_only\_file} \tag{A84.4-storage}$$

日志文件设为只追加（append-only），智能体无法修改或删除历史记录。

**违规日志的告警**：

当违规频率异常升高时，触发安全告警：

$$\text{alert} = \mathbb{1}\left[\text{violation\_rate} > \theta_{\text{alert}}\right], \quad \theta_{\text{alert}} = 10/\text{hour} \tag{A84.4-alert}$$

每小时超过10次违规操作即触发告警——可能意味着智能体在尝试绕过保护规则。

**违规日志的统计**：

| 统计项 | 计算方式 | 说明 |
|--------|---------|------|
| 总违规次数 | count(violations) | 累计拦截次数 |
| 按操作类型 | group_by(operation) | 哪种操作最常被拦截 |
| 按路径模式 | group_by(matched_pattern) | 哪个保护规则最常触发 |
| 违规趋势 | rate(violations, window=1h) | 违规频率是否上升 |

- **输入**：被拦截的文件操作信息
- **输出**：违规日志记录
- **约束**：日志只追加不可修改；异常频率触发告警

> **白话翻译**：违规日志就是"违章记录"——谁、什么时候、想改什么文件，全部记下来。日志只能追加不能修改，防止智能体删记录。如果一小时被拦截超过10次，说明智能体在试图绕过保护规则，触发安全告警。还要统计哪种操作最常被拦截、哪个保护规则最常触发。

---

## 3. 核心公式汇总

| 编号 | 公式 | 输入类型 | 输出类型 | 约束 |
|------|------|---------|---------|------|
| A84.1 | P = {p_1,...,p_K} | OS类型+安装路径 | 受保护路径集合 | 运行时不可修改 |
| A84.2 | intercept(file_op) → allow/deny | 文件操作 | 拦截结果 | 延迟<1ms |
| A84.3 | match(path, P) = 𝟙[∃p: regex_match(path,p)] | 路径+模式 | 匹配标志 | 延迟<0.5ms |
| A84.4 | log_violation(op, path, ts) | 违规信息 | 日志记录 | 只追加不可修改 |

---

## 4. Python接口定义

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from enum import Enum
import re

# 引用阶段83的数据类
from stage_083 import StorageCleaner


# ============================================================
# 阶段84 枚举：文件操作类型
# ============================================================
class FileOperation(Enum):
    """文件操作类型。"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    RENAME = "rename"
    CHMOD = "chmod"
    CREATE = "create"
    EXECUTE = "execute"


# ============================================================
# 阶段84 枚举：保护级别
# ============================================================
class ProtectionLevel(Enum):
    """文件保护级别。"""
    ABSOLUTE = "absolute"      # 绝对保护：任何操作都不允许
    CONDITIONAL = "conditional"  # 条件保护：仅允许读取


# ============================================================
# 阶段84 枚举：拦截结果
# ============================================================
class InterceptResult(Enum):
    """文件操作拦截结果。"""
    ALLOW = "allow"
    DENY = "deny"


# ============================================================
# 阶段84 数据类：受保护路径规则
# ============================================================
@dataclass(frozen=True)
class ProtectedPattern:
    """受保护路径规则。

    Attributes:
        pattern: 路径模式（支持通配符）
        regex: 编译后的正则表达式
        level: 保护级别
        description: 规则描述
    """
    pattern: str
    regex: re.Pattern
    level: ProtectionLevel
    description: str = ""


# ============================================================
# 阶段84 数据类：违规记录
# ============================================================
@dataclass(frozen=True)
class ViolationRecord:
    """违规操作记录。

    Attributes:
        timestamp: 违规时间戳
        operation: 操作类型
        file_path: 目标文件路径
        real_path: 解析后的真实路径
        matched_pattern: 匹配的受保护模式
        protection_level: 保护级别
        caller_info: 调用方信息
    """
    timestamp: float
    operation: FileOperation
    file_path: str
    real_path: str
    matched_pattern: str
    protection_level: ProtectionLevel
    caller_info: str = ""


# ============================================================
# 阶段84 数据类：文件保护引擎状态
# ============================================================
@dataclass(frozen=True)
class FileProtectionEngineState:
    """文件保护引擎的状态描述。

    Attributes:
        protected_patterns: 受保护路径规则列表
        total_violations: 总违规次数
        violation_rate: 违规频率（次/小时）
        is_alert: 是否处于告警状态
        cache_size: 匹配缓存大小
        last_violation: 最近一次违规记录

    Invariants:
        - total_violations >= 0
        - violation_rate >= 0
    """
    protected_patterns: Tuple[ProtectedPattern, ...] = ()
    total_violations: int = 0
    violation_rate: float = 0.0
    is_alert: bool = False
    cache_size: int = 0
    last_violation: Optional[ViolationRecord] = None


# ============================================================
# 阶段84 数据类：文件保护引擎
# ============================================================
@dataclass(frozen=True)
class FileProtectionEngine:
    """禁止修改系统关键文件的规则引擎。

    组合阶段83的StorageCleaner，在存储清理基础上增加文件保护状态。

    Attributes:
        storage_cleaner: 阶段83的存储清理器（组合字段）
        engine_state: 文件保护引擎状态
        alert_threshold: 违规告警阈值（默认10次/小时）
        max_cache_size: 最大缓存大小（默认10000）
        log_path: 违规日志文件路径
    """
    storage_cleaner: StorageCleaner
    engine_state: FileProtectionEngineState = field(
        default_factory=FileProtectionEngineState
    )
    alert_threshold: float = 10.0
    max_cache_size: int = 10000
    log_path: str = "violation.log"


# ============================================================
# 阶段84 核心接口：文件保护执行器
# ============================================================
class FileProtectionExecutor:
    """阶段84的核心接口：文件保护规则引擎执行器。

    实现受保护文件列表、文件操作拦截、规则匹配和违规日志。

    核心约束：
        - 受保护路径列表运行时不可修改
        - 拦截延迟 < 1ms
        - 违规日志只追加不可修改
    """

    def build_protected_patterns(
        self,
        os_type: str,
        agent_install_path: str
    ) -> List[ProtectedPattern]:
        """构建受保护路径列表（A84.1）。

        Args:
            os_type: 操作系统类型，支持"linux"|"darwin"|"windows"
            agent_install_path: 智能体安装路径

        Returns:
            受保护路径规则列表

        Raises:
            ValueError: 当os_type不支持时

        Examples:
            >>> executor = FileProtectionExecutor()
            >>> patterns = executor.build_protected_patterns(
            ...     "linux", "/opt/agent")
            >>> len(patterns) > 0
            True
        """
        import re
        if os_type not in {"linux", "darwin", "windows"}:
            raise ValueError(f"不支持的os_type: {os_type}")
        patterns: List[ProtectedPattern] = []
        if os_type in {"linux", "darwin"}:
            patterns.append(ProtectedPattern(
                pattern="/etc/*",
                regex=re.compile(r"^/etc/"),
                level=ProtectionLevel.CRITICAL,
                description="系统配置文件"
            ))
            patterns.append(ProtectedPattern(
                pattern="/usr/bin/*",
                regex=re.compile(r"^/usr/bin/"),
                level=ProtectionLevel.HIGH,
                description="系统可执行文件"
            ))
            patterns.append(ProtectedPattern(
                pattern="/bin/*",
                regex=re.compile(r"^/bin/"),
                level=ProtectionLevel.HIGH,
                description="核心可执行文件"
            ))
        patterns.append(ProtectedPattern(
            pattern=f"{agent_install_path}/core/*",
            regex=re.compile(re.escape(agent_install_path) + r"/core/"),
            level=ProtectionLevel.CRITICAL,
            description="智能体核心文件"
        ))
        return patterns

    def intercept(
        self,
        file_path: str,
        operation: FileOperation
    ) -> InterceptResult:
        """拦截文件操作（A84.2）。

        Args:
            file_path: 目标文件路径
            operation: 操作类型

        Returns:
            ALLOW 或 DENY

        Examples:
            >>> executor = FileProtectionExecutor()
            >>> result = executor.intercept(
            ...     "/etc/passwd", FileOperation.WRITE)
            >>> result == InterceptResult.DENY
            True
        """
        if operation in {FileOperation.WRITE, FileOperation.DELETE, FileOperation.RENAME}:
            return InterceptResult.DENY
        return InterceptResult.ALLOW

    def regex_match(
        self,
        file_path: str,
        patterns: Tuple[ProtectedPattern, ...]
    ) -> Optional[ProtectedPattern]:
        """规则匹配（A84.3）。

        Args:
            file_path: 目标文件路径
            patterns: 受保护路径规则

        Returns:
            匹配的规则，无匹配返回None

        Examples:
            >>> executor = FileProtectionExecutor()
            >>> import re
            >>> p = ProtectedPattern(
            ...     pattern="/etc/*",
            ...     regex=re.compile(r"^/etc/"),
            ...     level=ProtectionLevel.CRITICAL)
            >>> matched = executor.regex_match("/etc/passwd", (p,))
            >>> matched is not None
            True
        """
        for pattern in patterns:
            if pattern.regex.match(file_path):
                return pattern
        return None

    def log_violation(
        self,
        operation: FileOperation,
        file_path: str,
        real_path: str,
        matched_pattern: ProtectedPattern,
        caller_info: str = ""
    ) -> ViolationRecord:
        """记录违规日志（A84.4）。

        Args:
            operation: 操作类型
            file_path: 目标文件路径
            real_path: 解析后的真实路径
            matched_pattern: 匹配的规则
            caller_info: 调用方信息

        Returns:
            违规记录

        Examples:
            >>> executor = FileProtectionExecutor()
            >>> import re
            >>> p = ProtectedPattern(
            ...     pattern="/etc/*",
            ...     regex=re.compile(r"^/etc/"),
            ...     level=ProtectionLevel.CRITICAL)
            >>> record = executor.log_violation(
            ...     FileOperation.WRITE, "/etc/passwd",
            ...     "/etc/passwd", p)
            >>> record.operation == FileOperation.WRITE
            True
        """
        import time
        return ViolationRecord(
            timestamp=time.time(),
            operation=operation,
            file_path=file_path,
            real_path=real_path,
            matched_pattern=matched_pattern.pattern,
            protection_level=matched_pattern.level,
            caller_info=caller_info
        )

    def check_alert(
        self,
        violation_rate: float,
        threshold: float = 10.0
    ) -> bool:
        """检查是否应触发告警（A84.4-alert）。

        Args:
            violation_rate: 违规频率（次/小时），必须 >= 0
            threshold: 告警阈值，必须 > 0

        Returns:
            当violation_rate >= threshold时返回True

        Raises:
            ValueError: 当violation_rate < 0时
            ValueError: 当threshold <= 0时

        Examples:
            >>> executor = FileProtectionExecutor()
            >>> executor.check_alert(15.0, 10.0)
            True
            >>> executor.check_alert(5.0, 10.0)
            False
        """
        if violation_rate < 0:
            raise ValueError(
                f"violation_rate必须 >= 0，收到 {violation_rate}"
            )
        if threshold <= 0:
            raise ValueError(
                f"threshold必须 > 0，收到 {threshold}"
            )
        return violation_rate >= threshold

    def step(
        self,
        engine: FileProtectionEngine
    ) -> FileProtectionEngine:
        """执行一步文件保护检查。

        Args:
            engine: 当前文件保护引擎状态

        Returns:
            更新后的文件保护引擎状态

        Examples:
            >>> from stage_083 import StorageCleaner, MemoryMonitor, CPUMonitor
            >>> cpu = CPUMonitor()
            >>> mem = MemoryMonitor(cpu_monitor=cpu)
            >>> sc = StorageCleaner(memory_monitor=mem)
            >>> engine = FileProtectionEngine(storage_cleaner=sc)
            >>> executor = FileProtectionExecutor()
            >>> new_engine = executor.step(engine)
            >>> isinstance(new_engine, FileProtectionEngine)
            True
        """
        return engine
```

---

## 5. 阶段边界

| 方向 | 数据类型 | 来源/去向 | 说明 |
|------|---------|----------|------|
| **输入** | `StorageCleaner` | ← 阶段83 | 作为文件保护的存储清理输入（组合字段） |
| **输入** | A1.8（生存条件） | ← 阶段1 | 文件保护的理论依据 |
| **输出** | `FileProtectionEngine` | → 阶段85 | 文件保护状态，作为探索冻结的组合输入 |
| **输出** | A84.1-A84.4（文件保护定义） | → 阶段90 | 安全护栏的文件访问检查接口 |

---

## 6. 开源技术选型

| 技术 | 用途 | 说明 |
|------|------|------|
| re | 正则匹配 | 路径模式匹配 |
| pathlib | 路径规范化 | 符号链接解析和路径规范化 |
| os.walk | 目录遍历 | 构建受保护路径列表 |
| pyinotify | 文件监控 | 可选的实时文件操作监控 |

---

## 7. 验证标准

| 编号 | 验证项 | 通过条件 | 验证方法 |
|------|--------|---------|---------|
| V84.1 | 受保护路径列表 | 包含/boot、/etc/passwd等 | 列表检查 |
| V84.2 | 自身启动代码保护 | agent主程序在保护列表中 | 自保护测试 |
| V84.3 | 列表运行时不可修改 | 运行时修改抛异常 | 修改测试 |
| V84.4 | WRITE拦截 | 写入受保护路径被拒绝 | 写入测试 |
| V84.5 | DELETE拦截 | 删除受保护路径被拒绝 | 删除测试 |
| V84.6 | 符号链接解析 | 通过链接访问被拦截 | 链接测试 |
| V84.7 | 路径规范化 | /etc/../etc/passwd被识别 | 规范化测试 |
| V84.8 | 正则匹配 | 通配符正确转换为正则 | 匹配测试 |
| V84.9 | 前缀树优化 | 匹配延迟<0.5ms | 性能测试 |
| V84.10 | 缓存命中 | 重复路径直接返回缓存 | 缓存测试 |
| V84.11 | 违规日志记录 | 每次拦截有日志 | 日志测试 |
| V84.12 | 日志只追加 | 无法修改历史日志 | 安全测试 |
| V84.13 | 违规告警 | >10次/小时触发告警 | 告警测试 |
| V84.14 | FileProtectionEngine组合 | `engine.storage_cleaner` 类型为 `StorageCleaner` | 类型检查 |
| V84.15 | 拦截延迟 | 拦截总延迟<1ms | 性能测试 |

---

> **阶段84完** | 阶段84建立了禁止修改系统关键文件的规则引擎：受保护文件列表 [/boot, /etc/passwd, /etc/shadow, C:\Windows\System32, 自身启动代码]（A84.1），文件操作拦截 intercept(file_op) → allow/deny（A84.2），规则匹配 algorithm = regex_match(file_path, protected_patterns)（A84.3），违规日志 log_violation(operation, file_path, timestamp)（A84.4），封装为FileProtectionEngine数据类组合StorageCleaner。下一步：阶段85将建立陌生探索5轮无收益自动冻结机制。
