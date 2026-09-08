# from tools.embedding import find_nearest_label

# prompt = "xây dựng hệ thống dạy hóa học trực tuyến"

# find_nearest_label(prompt)

from engine.pipeline import AsyncPipelineEngine, AsyncToolRegistry
import preprocessing


# @registry.registry()
# async def preprocessing

async def main():
    registry = AsyncToolRegistry()
    registry.tools['preprocessing'] = preprocessing.preprocessing
    ape = AsyncPipelineEngine(registry, max_concurrent_tasks=5)
    ape.add_step("preprocessing", "preprocessing")
    result = await ape.run()

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())