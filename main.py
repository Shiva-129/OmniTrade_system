from src.core.engine import TradingEngine


def main():
    engine = TradingEngine()
    try:
        import asyncio
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
