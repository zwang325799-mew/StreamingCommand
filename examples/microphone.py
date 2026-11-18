import asyncio
from streamingcommand import StreamingCommand


async def main():
    async with StreamingCommand() as recognizer:
        async for event in recognizer.listen():
            if event["type"] in ("transcript.final", "command.final", "command.error"):
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
