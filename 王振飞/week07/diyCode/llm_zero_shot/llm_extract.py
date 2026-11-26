"""LLM 零样本/少样本：构造提示、调用兼容接口、解析并校验 JSON。"""
import json, os, re, urllib.error, urllib.request
from common.event_data import ROOT, validate
EXAMPLES = [{'text': '甲公司拟回购股份。', 'events': [{'type': '股份回购', 'trigger': {'text': '回购'}, 'arguments': [{'role': '回购方', 'text': '甲公司'}]}]}]

def load_config():
    """每次请求读取本地配置；配置文件非空时优先，空项回退到环境变量。"""
    path = ROOT / 'llm_config.json'
    config = {}
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError('llm_config.json 读取失败，请检查 JSON 格式') from exc
        if not isinstance(config, dict):
            raise ValueError('llm_config.json 必须是 JSON 对象')
    values = {}
    for field, env in [('api_key', 'LLM_API_KEY'), ('base_url', 'LLM_BASE_URL'), ('model', 'LLM_MODEL')]:
        value = config.get(field, '') or os.environ.get(env, '')
        if not isinstance(value, str):
            raise ValueError(f'LLM 配置项 {field} 必须是字符串')
        values[field] = value.strip()
    if not all(values.values()):
        raise ValueError('请在 diyCode/llm_config.json 中填写 api_key、base_url 和 model')
    values['base_url'] = values['base_url'].rstrip('/')
    return values


def extract(text, schema, mode):
    """zero 不提供示例；few 在消息中加入示例。两者均无需训练。"""
    config = load_config()
    key, base, model = config['api_key'], config['base_url'], config['model']
    # 给出允许的事件类型和角色，要求模型只摘取原文片段。
    system = '你是金融公告事件抽取助手。仅输出 JSON 对象，格式为 {"events":[{"type":事件类型,"trigger":{"text":原文触发词},"arguments":[{"role":角色,"text":原文片段}]}]}。没有事件返回空数组。只能使用给定类型及对应角色，不能编造原文。类型和角色：' + json.dumps(schema, ensure_ascii=False)
    messages = [{'role': 'system', 'content': system}]
    # few-shot 将示例作为一组历史问答，帮助模型理解输出格式。
    if mode == 'few':
        for ex in EXAMPLES:
            messages.extend([{'role': 'user', 'content': ex['text']}, {'role': 'assistant', 'content': json.dumps({'events': ex['events']}, ensure_ascii=False)}])
    messages.append({'role': 'user', 'content': text})
    payload = json.dumps({'model': model, 'messages': messages, 'temperature': 0}, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(base + '/chat/completions', data=payload, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'模型接口请求失败（HTTP {e.code}），请检查配置和服务商状态') from e
    except urllib.error.URLError as e:
        raise RuntimeError('无法连接模型接口，请检查地址和网络') from e
    try:
        raw = result['choices'][0]['message']['content'].strip()
        raw = re.sub('^```(?:json)?\\s*|\\s*```$', '', raw, flags=re.I)
        obj = json.loads(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get('events'), list):
            raise ValueError('missing events')
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError('模型返回的内容不是预期 JSON，请重试') from e
    # 不信任模型生成的位置，统一在原文中定位并检查类型、角色。
    return validate(text, obj['events'], schema)
