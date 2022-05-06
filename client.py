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

    def connection_made(self, transport):
        print("Connected to server!")


async def main():
    loop = asyncio.get_running_loop()
    on_con_lost = loop.create_future()
    SERVER_SETTINGS = ("localhost", 8052)
    print(f"Connecting to socket on: {SERVER_SETTINGS}")
    transport, protocol = await loop.create_connection(
        lambda: EchoClientProtocol("Echo subscription!", on_con_lost), *SERVER_SETTINGS
    )
    try:
        await on_con_lost
    finally:
        transport.close()


asyncio.run(main())
