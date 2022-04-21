import asyncio


class EchoClientProtocol(asyncio.Protocol):
    def __init__(self, message, on_con_lost):
        self.message = message
        self.on_con_lost = on_con_lost

    def data_received(self, data):
        print("Data received: {!r}".format(data.decode()))

    def connection_lost(self, exc):
        print("The server closed the connection")
        self.on_con_lost.set_result(True)


async def main():
    loop = asyncio.get_running_loop()
    on_con_lost = loop.create_future()
    message = "Echo subscription!"

    transport, protocol = await loop.create_connection(
        lambda: EchoClientProtocol(message, on_con_lost), "127.0.0.1", 8052
    )

    try:
        await on_con_lost
    finally:
        transport.close()


asyncio.run(main())
