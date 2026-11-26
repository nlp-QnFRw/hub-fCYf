# LLM Zero-shot / Few-shot

`llm_extract.py` 使用 OpenAI 兼容的 chat completions 接口，流程是读取配置 → 构造 schema 和示例 → 调用模型 → 解析 JSON → 公共校验。无训练或本地模型下载。

## 配置

在 `diyCode/llm_config.json` 填写 `base_url`、`model`、`api_key`；没有该文件时可复制 `llm_config.example.json`。配置中非空值优先，空项才回退对应 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 环境变量。每次请求都会读取文件，因此只修改配置后下一次请求即可生效。不要把 Key 提交或发送给别人。

## 两种提示方式

- Zero-shot：发送任务说明、允许的事件类型与角色、用户输入，不给示例。
- Few-shot：额外发送 `EXAMPLES` 中一个固定“股份回购”示例。它教模型参考输出格式，并不会训练或更新模型；其示例未覆盖所有 13 类事件。

程序要求模型输出原文片段，再由公共模块检查类型、角色及原文位置。无法解析 JSON 会报错；输出为空不等于请求失败。输入内容发送给配置的服务商，调用可能产生费用。

401 时检查 Key、API 根地址、服务商与模型配置，不要打印密钥。若修改过 Python 代码，要重启服务；修改配置文件本身无需重启。
