# Python Code Review 专项：代码质量管理 + 代码审查落地

## 一、Python 传统代码质量完整落地（无AI前置基线）

### 1. 编码规范 & 本地前置拦截（pre-commit）

#### 统一规范工具栈

- 格式化：`black`（强制代码风格）
- 导入排序：`isort`
- 静态Lint：`ruff`（替代flake8/pylint，速度更快）
- 类型检查：`mypy`（Python质量核心，杜绝动态类型坑）
- 文档规范：`pydocstyle` + `interrogate`（检查缺失docstring）
- 密钥/敏感信息扫描：`detect-secrets`

#### pre-commit 强制钩子（项目根目录`.pre-commit-config.yaml`）

提交代码前自动执行，不通过禁止git commit：

1. black + isort 格式化
2. ruff 代码异味、语法、复杂度扫描
3. mypy 类型校验（生产项目必须开启严格模式 `strict = true`）
4. detect-secrets 扫描硬编码密钥、密码、token
5. interrogate 检查函数/类无注释
6. bandit Python安全漏洞扫描（Python专属OWASP工具）

#### 编码基线强制约束

1. 全部业务代码强制类型注解：`def func(x: int) -> str`，禁止大量`Any`
2. 单行长度120，函数最大行数50，嵌套层级≤4
3. 禁止`eval/exec/os.system`无过滤拼接字符串
4. 禁止裸except，必须捕获具体异常
5. 数据库SQL统一使用ORM参数化查询，禁止f-string拼接SQL
6. 日志禁止打印手机号、身份证、明文密码等敏感数据

### 2. CI流水线自动化质量门禁（Python专属卡点）

流水线MR/PR合并前全部阻断，红灯不允许合入：

| 检查项 | Python工具 | 阻断红线标准 |
|--------|------------|--------------|
| 静态安全扫描 | bandit | 高危漏洞（命令注入、SQL拼接、不安全pickle等）直接阻断 |
| 代码复杂度 | ruff（cyclomatic-complexity） | 圈复杂度>12阻断 |
| 类型检查 | mypy --strict | 类型报错零容忍，禁止临时忽略 |
| 单元测试 | pytest + coverage.py | 新增代码覆盖率≥90%，整体≥80%；无失败case |
| 依赖漏洞 | safety / pip-audit | 依赖高危漏洞直接阻断；requirements.txt/pyproject.toml统一管理 |
| 重复代码 | ruff / radon | 重复代码块>15行标记整改 |
| 打包规范 | poetry/pipenv | 禁止手动install包，新增依赖必须写入配置文件 |

补充Python特有风险拦截：

- 禁止`pickle`反序列化外部不可信数据
- 禁止使用`yaml.unsafe_load()`，强制`safe_load`
- 校验文件路径防止路径穿越（`pathlib`规范使用）

### 3. Python 代码审查 Code Review 标准清单

#### （1）类型与健壮性（Python重点）

- 是否加完整类型注解，复杂对象是否用`TypedDict/dataclass/Pydantic`
- 无裸`except:`，异常分级捕获，有重试/降级逻辑
- 入参校验：接口/函数入参用Pydantic模型校验，不手动if判断
- 资源释放：文件、数据库连接、HTTP会话是否`with`上下文管理

#### （2）安全（Python高频漏洞）

- SQL是否全部参数化，无f-string拼接SQL
- shell调用是否使用subprocess传参数列表，而非字符串拼接
- yaml/json加载是否使用安全方法
- 密钥、数据库连接串是否存在环境变量，无硬编码
- 用户输入是否过滤，防止XSS、路径遍历

#### （3）性能与Python特性坑

- 循环内IO、数据库查询（N+1问题）
- 列表循环频繁append优先预分配，避免大量临时对象
- 大文件读取不用read()一次性加载，流式读取
- 异步代码：async/await是否正确，有无阻塞IO同步调用
- 避免全局变量、可变默认参数陷阱 `def f(a=[]):`

#### （4）工程规范

- 分层清晰：接口层、service业务层、dao数据层隔离
- 常量统一抽离，无魔法数字/魔法字符串
- 工具函数抽离，无重复CRUD复制粘贴
- 每个公共方法、类必须docstring，说明入参、返回、异常

#### （5）测试

- 正常流程、异常、边界值全覆盖
- 数据库操作mock，不依赖测试环境真实库
- 工具类、核心算法单独写单元测试

## 二、Python Vibe Coding 特有风险（AI生成Python代码痛点）

Vibe Coding：用自然语言让AI批量生成Python脚本、接口、爬虫、数据处理脚本，纯靠AI输出，人少思考，Python下独有的问题：

1. **类型注解缺失**：AI经常省略mypy类型，全是`Any`，运行时报错
2. **危险API滥用**：随意生成`eval、exec、pickle.load、os.system、yaml.unsafe_load`
3. **SQL拼接漏洞**：大量用f-string拼接SQL，极易注入
4. **资源泄漏**：忘记`with open()`、数据库连接不关闭，无上下文管理器
5. **异常处理缺失**：裸except、完全不捕获异常，生产直接崩溃
6. **依赖混乱**：随机引入冷门第三方库，不写入requirements/poetry，本地能跑线上缺失包
7. **数据处理隐患**：Pandas/NumPy代码不处理空值、类型转换异常
8. **幻觉逻辑**：编造不存在数据库表、接口函数、第三方库API
9. **无单测**：AI默认不写pytest用例，覆盖率归零
10. 异步代码错误混用：同步阻塞代码塞进async函数，性能卡死

## 三、Python 场景下 Vibe Coding 完整管控方案

核心思路：**给AI强制Python专属规则 + 本地实时校验 + CI强化Python安全扫描 + CR增加AI代码专项检查**

### 1. 生成前：约束AI输出Python代码（项目统一AI规则文件）

项目根目录新建 `.cursorrules` / `ai_rules.py`，AI读取后强制遵循：

```
# Python生成强制规则
1. 所有函数、类、返回值必须完整类型注解，复杂结构使用Pydantic/TypedDict/dataclass
2. 禁止使用eval、exec、pickle、yaml.unsafe_load、os.system字符串拼接
3. SQL语句全部使用参数化查询，严禁f-string拼接SQL
4. 文件、数据库、http连接必须使用with上下文管理器自动释放资源
5. 异常禁止裸except，捕获具体Exception，增加日志与降级处理
6. 新增任何第三方库必须输出对应poetry add/requirements条目
7. 生成业务代码同步输出完整pytest单元测试，覆盖正常/异常/边界
8. 函数行数不超过50行，超过自动拆分工具函数
9. 公共类、方法必须写google风格docstring
10. 禁止可变参数默认值，禁止全局可变变量
11. 日志脱敏，敏感数据不打印
12. 异步代码区分async/同步，不混用阻塞IO
```

#### 标准化Prompt模板



???+ note "标准化Prompt模板"

    每次AI生成代码必须带上：
    
    基于Python3.10+，遵循项目black+ruff+mypy规范；严格遵循ai_rules.py约束；输出代码后附带：

    1. 代码风险自查报告（安全、性能、类型、异常）
    2. 配套pytest单元测试用例
    3. 新增依赖清单
    4. 代码使用示例

高风险模块（数据入库、支付、爬虫、文件解析、权限校验）禁止纯Vibe Coding，必须先人工设计逻辑，AI仅辅助实现。

### 2. 生成中：IDE实时拦截AI劣质Python代码

1. IDE插件实时开启：
   - Ruff Linter、Mypy、Bandit实时标红危险代码
   - 发现`eval、pickle、f-string SQL`直接高亮警告
2. AI自审流程：生成代码后，让大模型基于bandit/ruff规则自查，自动修复类型、安全缺陷
3. 禁止在AI对话粘贴数据库账号、生产数据表、真实业务数据，统一使用模拟数据

### 3. CI流水线增强：AI生成Python代码专项检测

在原有Python质量门禁基础上增加3步AI专属校验：

1. **AI代码标记**：识别PR中AI生成代码块，提升扫描严格等级
2. Bandit深度扫描：对AI代码启用全部高危规则，发现不安全函数直接阻断
3. Mypy严格模式强制：AI代码不允许临时`# type: ignore`跳过类型检查
4. 依赖校验：AI新增包必须人工评审，pip-audit扫描漏洞
5. 测试加码：AI生成代码新增覆盖率必须100%，否则阻断合并

### 4. Python CR 新增Vibe Coding专项审查清单（必查阻断项）

评审人优先检查AI最容易出错的Python特有问题：

1. 【幻觉校验】数据库表、字段、第三方函数是否真实存在，AI有没有编造接口
2. 【类型安全】是否完整类型注解，无大量Any，mypy无报错
3. 【高危函数】是否存在eval/exec/pickle/unsafe yaml/拼接shell/SQL
4. 【资源管理】文件、DB、HTTP会话是否用with自动关闭
5. 【异常处理】无裸except，边界、超时、空值全部处理
6. 【依赖管理】新增库是否写入poetry/requirements，无遗漏
7. 【测试完备】pytest可本地运行，覆盖异常场景
8. 【Python陷阱】可变默认参数、全局变量、异步阻塞等问题
9. 【敏感数据】日志、返回值是否脱敏，无硬编码密钥

分层评审策略：

- 简单工具脚本、CRUD：自动化全绿后1人快速复核
- 业务接口、数据处理（Pandas/ETL）：资深开发逐条审核
- 资金、鉴权、文件解析、爬虫：双人复核+技术负责人终审

### 5. 配套团队管理制度（Python AI编码约束）

1. 原型脚本（本地测试）：可宽松Vibe Coding，但禁止提交生产分支
2. 生产Python代码：不允许AI代码不经pre-commit+CR直接合并
3. 每月抽取线上AI生成Python代码，用bandit+mypy做专项安全审计
4. 高频AI缺陷统一更新到`.cursorrules`，减少重复踩坑

## 四、Python完整工具栈（本地+CI+AI评审）

### 本地开发（Vibe Coding实时校验）

- AI编码：Cursor、GitHub Copilot、Claude Code
- 格式化/导入：black, isort
- Lint/复杂度：ruff
- 类型检查：mypy
- 安全扫描：bandit
- 密钥扫描：detect-secrets
- 单测：pytest + coverage
- 前置钩子：pre-commit

### CI流水线自动化

- 依赖漏洞：pip-audit、safety
- 项目依赖管理：poetry
- 代码度量：radon（圈复杂度）
- AI辅助评审机器人：CodeRabbit、通义灵码Python专项扫描

### 线上审计

- 批量代码安全扫描：bandit批量执行
- 类型全量校验：mypy --strict

## 五、极简落地执行步骤（Python团队可直接落地）

1. 统一项目`pyproject.toml`，配置black/ruff/mypy/bandit标准规则；
2. 配置pre-commit，强制本地提交前全量校验；
3. CI流水线搭建Python质量门禁，阻断类型、安全、测试不达标PR；
4. 编写Python专属AI规则文件+标准化Prompt，约束AI生成代码；
5. 升级CR清单，增加AI Python代码专项检查项；
6. 区分原型/生产代码，高风险模块限制纯Vibe Coding；
7. 月度专项审计AI产出Python代码，持续优化规则。

## 核心总结（Python场景关键差异）

Python动态语言天生缺少编译期校验，AI生成代码极易埋下类型、安全、资源泄漏隐患；
因此Vibe Coding下必须**强化mypy静态类型+bandit安全扫描**两道自动化防线，人工CR重点盯AI高频踩坑的Python特有陷阱，自动化拦截低级错误，人把控业务逻辑、数据安全与架构合理性。