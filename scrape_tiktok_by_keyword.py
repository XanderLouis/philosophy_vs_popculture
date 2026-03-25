import asyncio
from TikTokApi import TikTokApi
import pandas as pd
from pathlib import Path
import time

# Your target keywords
keywords = ["freedom", "meaning", "identity", "love", "truth", "death", "vibe", "power"]
MAX_VIDEOS = 10
MAX_COMMENTS = 100

# Where to save
output_dir = Path("data/pop_texts/tiktok_api")
output_dir.mkdir(parents=True, exist_ok=True)

async def scrape_trending_and_filter(api, keyword):
    results = []
    print(f"\n🔍 Searching trending TikToks for keyword: '{keyword}'")

    try:
        # Pull a broader range of trending videos
        videos = await api.trending(count=40)

        async for video in videos:
            try:
                data = video.as_dict
                caption = data.get("desc", "")

                # Filter based on keyword in caption
                if keyword.lower() not in caption.lower():
                    continue

                video_url = f"https://www.tiktok.com/@{data['author']['uniqueId']}/video/{data['id']}"

                # Get comments
                comments = []
                async for comment in api.video.comments(video_id=data["id"]):
                    comments.append(comment.get("text", ""))
                    if len(comments) >= MAX_COMMENTS:
                        break

                results.append({
                    "keyword": keyword,
                    "video_url": video_url,
                    "caption": caption,
                    "comments": " ||| ".join(comments)
                })
                print(f"✅ Captured: {video_url}")

            except Exception as e:
                print(f"⚠️ Error on video: {e}")
                continue

    except Exception as e:
        print(f"❌ Couldn’t fetch trending videos: {e}")

    return results

async def main():
    all_data = []
    async with TikTokApi() as api:
        for kw in keywords:
            filtered = await scrape_trending_and_filter(api, kw)
            all_data.extend(filtered)
            time.sleep(2)

    df = pd.DataFrame(all_data)
    df.to_csv(output_dir / "tiktokapi_filtered_trending.csv", index=False, encoding='utf-8')
    print("\n✅ Scraping complete. Data saved.")

if __name__ == "__main__":
    asyncio.run(main())
