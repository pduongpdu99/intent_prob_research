# from tools.embedding import find_nearest_label

# prompt = "xây dựng hệ thống dạy hóa học trực tuyến"

# find_nearest_label(prompt)

from engine.pipeline import AsyncPipelineEngine, AsyncToolRegistry
import preprocessing


async def main():
    registry = AsyncToolRegistry()
    registry.tools['initialization'] = preprocessing.preprocessing
    registry.tools['template'] = preprocessing.export_template
    registry.tools['klb'] = preprocessing.export_klb
    registry.tools['SYNC'] = preprocessing.sync_node

    ape = AsyncPipelineEngine(registry, max_concurrent_tasks=5)
    ape.add_step("START", tool_name="initialization")
    ape.add_step("A", tool_name="template", depends_on=['START'])
    ape.add_step("B", tool_name="klb", depends_on=['START'])
    ape.add_step("END", tool_name="SYNC", depends_on=['A', "B"])

    result = await ape.run()

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())