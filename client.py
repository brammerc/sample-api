import sys
import asyncio
import time

from numpy import percentile, mean
from pandas import DataFrame, read_csv
from httpx import AsyncClient


# Update if running elsewhere (e.g., ECS, different host/port)
BASE_URL = "http://localhost:8080"
ENDPOINT = "/predict"

USERNAME = "admin"
PASSWORD = "changeme"


def load_input_csv(csv_path: str) -> DataFrame:
    df = read_csv(csv_path)

    expected_columns = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"
    ]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    return df


async def send_prediction_request(client: AsyncClient, payload: dict, semaphore: asyncio.Semaphore):
    # Semaphore prevents overwhelming the OS/server with too many simultaneous connections
    async with semaphore:
        start = time.perf_counter() # Start client-side timer
        try:
            response = await client.post(
                f"{BASE_URL}{ENDPOINT}",
                json=payload,
                auth=(USERNAME, PASSWORD),
                timeout=10.0
            )
            # Capture total client-side time in ms
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # If the server returned an error (4xx or 5xx)
            if response.status_code != 200:
                return {
                    "error": True,
                    "status_code": response.status_code,
                    "reason": response.text,  # This will show the FastAPI error detail
                    "client_elapsed_ms": elapsed_ms
                }

            data = response.json()
            data["client_elapsed_ms"] = elapsed_ms
            return data

        except Exception as e:
            # Captures connection errors (e.g., wrong URL or server down)
            return {
                "error": True,
                "status_code": "CONNECTION_FAILURE",
                "reason": str(e),
                "client_elapsed_ms": (time.perf_counter() - start) * 1000
            }


async def main():
    if len(sys.argv) != 2:
        print("Usage: python client.py input.csv")
        sys.exit(1)

    df = load_input_csv(sys.argv[1])
    payloads = df.to_dict(orient="records")
    
    # Limits concurrent requests (e.g., 50 at a time) to stay within server limits
    semaphore = asyncio.Semaphore(50)
    
    start_time = time.perf_counter()

    async with AsyncClient() as client:
        tasks = [send_prediction_request(client, p, semaphore) for p in payloads]
        results = await asyncio.gather(*tasks)


    total_time = time.perf_counter() - start_time
    
    # Process results
    successes = [r for r in results if "prediction" in r]
    failures = [r for r in results if r.get("metadata", {}).get("status") == "error"]
    latencies = [r.get("metadata").get("latency_ms", 0) for r in successes]
    total_requests = len(results)
    success_count = len(successes)
    error_count = total_requests - success_count

    # Print the "First 5" for visual verification
    print("\n--- SAMPLE DATA (First 5 Successes) ---")
    for i, res in enumerate(successes[:5], 1):
        print(f"{i}. Prediction: ${res['prediction']:,.2f} | Timestamp: {res['metadata']['prediction_timestamp']} UTC")


    # Print the first 3 failures to identify the cause
    if failures:
        print(f"DEBUG: Found {len(failures)} items in the failures list.")
        print("\n--- ERROR DIAGNOSTICS (First 3 Failures) ---")
        for i, err in enumerate(failures[:3], 1):
            print(f"Failure {i}:")
            print(f"  - Error Reason:      {err.get('error')}")

    # Performance Summary
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Total Requests: {total_requests}")  
    print(f"Total Success:  {success_count}")
    print(f"Total Failures: {error_count}")
    print(f"Error Rate:     {(error_count / total_requests) * 100:.2f}%")

    if success_count > 0:
        overheads = [r["client_elapsed_ms"] - r["metadata"]["latency_ms"] for r in successes]
        print(f"Avg Network Overhead: {mean(overheads):.2f} ms")
        print(f"P99 Network Overhead: {percentile(overheads, 99):.2f} ms")

        print(f"Throughput:    {total_requests / total_time:.2f} requests/sec")
        print(f"Average Server-side Processing Time:  {mean(latencies):.2f} ms")
        print(f"P99 (Worst):   {percentile(latencies, 99):.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
