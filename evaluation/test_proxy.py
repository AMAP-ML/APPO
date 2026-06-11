"""
测试 your-proxy.example.com 网关的连通性、响应时间和 QPS 限制
用法：python test_proxy.py
"""

import time
import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PROXY_URL = os.environ.get("BRIGHTDATA_PROXY_URL", "http://your-proxy.example.com/request")
API_KEY = os.environ.get("BRIGHTDATA_API_KEY", "")
ZONE = os.environ.get("BRIGHTDATA_ZONE", "your_zone")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-Dst-Host": "api.brightdata.com",
    "X-Dst-Scheme": "https",
}

TEST_QUERIES = [
    "python programming language",
    "machine learning basics",
    "deep learning neural network",
    "natural language processing",
    "reinforcement learning policy",
]


def single_request(query: str, timeout: int = 30) -> dict:
    """发送单个请求并返回结果"""
    payload = {
        "zone": ZONE,
        "url": f"https://www.bing.com/search?q={query.replace(' ', '+')}&brd_json=1&cc=cn",
        "format": "raw",
    }
    start_time = time.time()
    try:
        response = requests.post(PROXY_URL, headers=HEADERS, json=payload, timeout=timeout)
        elapsed = time.time() - start_time
        return {
            "query": query,
            "status": response.status_code,
            "elapsed": elapsed,
            "success": response.status_code == 200,
            "body_preview": repr(response.text[:100]),
            "error": None,
        }
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {
            "query": query,
            "status": None,
            "elapsed": elapsed,
            "success": False,
            "body_preview": None,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as exception:
        elapsed = time.time() - start_time
        return {
            "query": query,
            "status": None,
            "elapsed": elapsed,
            "success": False,
            "body_preview": None,
            "error": str(exception),
        }


def test_single_connectivity():
    """测试1：单次请求连通性"""
    print("\n" + "=" * 60)
    print("测试1：单次请求连通性")
    print("=" * 60)
    result = single_request("hello world test", timeout=60)
    print(f"  状态码: {result['status']}")
    print(f"  耗时:   {result['elapsed']:.2f}s")
    print(f"  成功:   {result['success']}")
    print(f"  响应预览: {result['body_preview']}")
    if result['error']:
        print(f"  错误:   {result['error']}")
    return result['success']


def test_sequential_requests(count: int = 5):
    """测试2：顺序发送多个请求，测试稳定性和平均响应时间"""
    print("\n" + "=" * 60)
    print(f"测试2：顺序发送 {count} 个请求（测试稳定性）")
    print("=" * 60)
    results = []
    for i, query in enumerate(TEST_QUERIES[:count]):
        print(f"  [{i+1}/{count}] 发送请求: {query}")
        result = single_request(query, timeout=60)
        results.append(result)
        status_str = f"✅ {result['status']}" if result['success'] else f"❌ {result['status'] or result['error']}"
        print(f"         {status_str}, 耗时: {result['elapsed']:.2f}s")
        time.sleep(0.5)  # 顺序请求间隔 0.5s

    success_count = sum(1 for r in results if r['success'])
    avg_elapsed = sum(r['elapsed'] for r in results) / len(results)
    print(f"\n  成功率: {success_count}/{count}")
    print(f"  平均耗时: {avg_elapsed:.2f}s")
    return results


def test_concurrent_requests(concurrency: int = 5):
    """测试3：并发请求，测试 QPS 限制"""
    print("\n" + "=" * 60)
    print(f"测试3：并发 {concurrency} 个请求（测试 QPS 限制）")
    print("=" * 60)
    queries = [f"concurrent test query {i}" for i in range(concurrency)]
    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(single_request, query, 60): query for query in queries}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status_str = f"✅ {result['status']}" if result['success'] else f"❌ {result['status'] or result['error']}"
            print(f"  {result['query'][:30]}: {status_str}, 耗时: {result['elapsed']:.2f}s")

    total_elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    print(f"\n  总耗时: {total_elapsed:.2f}s")
    print(f"  成功率: {success_count}/{concurrency}")
    print(f"  实际 QPS: {concurrency / total_elapsed:.2f}")
    return results


def test_rapid_fire(count: int = 10, interval: float = 0.1):
    """测试4：快速连续请求，测试是否有 QPS 限制"""
    print("\n" + "=" * 60)
    print(f"测试4：快速连续 {count} 个请求（间隔 {interval}s，测试 QPS 上限）")
    print("=" * 60)
    results = []
    for i in range(count):
        query = f"rapid fire test {i}"
        result = single_request(query, timeout=30)
        results.append(result)
        status_str = f"✅ {result['status']}" if result['success'] else f"❌ {result['status'] or result['error']}"
        print(f"  [{i+1:2d}/{count}] {status_str}, 耗时: {result['elapsed']:.2f}s")
        time.sleep(interval)

    success_count = sum(1 for r in results if r['success'])
    timeout_count = sum(1 for r in results if r['error'] and 'Timeout' in str(r['error']))
    status_504_count = sum(1 for r in results if r['status'] == 504)
    print(f"\n  成功率: {success_count}/{count}")
    print(f"  504超时: {status_504_count}")
    print(f"  请求超时: {timeout_count}")
    return results


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set BRIGHTDATA_API_KEY before running this test.")
    print(f"🔍 开始测试代理网关: {PROXY_URL}")
    print("   API Key: <redacted>")
    print(f"   Zone: {ZONE}")

    # 测试1：单次连通性
    connected = test_single_connectivity()
    if not connected:
        print("\n❌ 网关连通性测试失败，请检查网关地址和 API Key")
    else:
        print("\n✅ 网关连通，继续后续测试...")

        # 测试2：顺序请求稳定性
        test_sequential_requests(count=5)

        # 测试3：并发请求 QPS
        test_concurrent_requests(concurrency=5)

        # 测试4：快速连续请求
        test_rapid_fire(count=10, interval=0.2)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
