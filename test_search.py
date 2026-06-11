"""
快速测试 Brightdata 搜索工具是否能通过公司代理网关正常工作。
运行方式：python /home/wangxucong.wxc/AERPO/ARPO/test_search.py
"""
import sys
import os
sys.path.iqnsert(0, "/home/wangxucong.wxc/AERPO/ARPO/verl_arpo_entropy")

import requests
from urllib.parse import urlencode

PROXY_BASE_URL = os.environ.get("BRIGHTDATA_PROXY_BASE_URL", "http://your-proxy.example.com")
API_KEY = os.environ.get("BRIGHTDATA_API_KEY", "")
ZONE = os.environ.get("BRIGHTDATA_ZONE", "your_zone")

def test_via_proxy(query: str = "Who directed the film Titanic"):
    print(f"=== 测试通过代理网关搜索 ===")
    print(f"代理地址: {PROXY_BASE_URL}/request")
    print(f"查询: {query}")

    encoded_query = urlencode({"q": query, "mkt": "en-US", "setLang": "en"})
    target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc=cn"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Dst-Host": "api.brightdata.com",
        "X-Dst-Scheme": "https",
    }
    payload = {
        "zone": ZONE,
        "url": target_url,
        "format": "raw"
    }

    try:
        proxy_url = f"{PROXY_BASE_URL}/request"
        print(f"\n发送 POST 请求到: {proxy_url}")
        print(f"Headers: {headers}")
        print(f"Payload: {payload}")

        response = requests.post(proxy_url, headers=headers, json=payload, timeout=30)

        print(f"\n=== 响应结果 ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Body Length: {len(response.text)}")
        print(f"Body Preview (前500字符): {repr(response.text[:500])}")

        if response.status_code == 200 and response.text.strip():
            import json
            try:
                data = json.loads(response.text)
                print(f"\n✅ JSON 解析成功！顶层 keys: {list(data.keys())}")
                if "organic" in data:
                    print(f"搜索结果数量: {len(data['organic'])}")
                    print(f"第一条结果: {data['organic'][0].get('description', '')[:200]}")
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON 解析失败: {e}")
        else:
            print(f"\n❌ 请求失败或响应为空")

    except Exception as e:
        print(f"\n❌ 请求异常: {type(e).__name__}: {e}")


def test_direct_brightdata(query: str = "Who directed the film Titanic"):
    """直接访问 Brightdata（不走代理），用于对比"""
    print(f"\n=== 测试直连 Brightdata（对比用）===")

    encoded_query = urlencode({"q": query, "mkt": "en-US", "setLang": "en"})
    target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc=cn"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "zone": ZONE,
        "url": target_url,
        "format": "raw"
    }

    try:
        response = requests.post(
            "https://api.brightdata.com/request",
            headers=headers,
            json=payload,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        print(f"Body Length: {len(response.text)}")
        print(f"Body Preview: {repr(response.text[:300])}")
    except Exception as e:
        print(f"❌ 直连失败（预期内）: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set BRIGHTDATA_API_KEY before running this test.")
    test_via_proxy()
    test_direct_brightdata()
