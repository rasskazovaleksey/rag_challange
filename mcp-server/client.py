import asyncio
import json
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from contextlib import AsyncExitStack

from openai import AsyncOpenAI


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = "gpt-4o"
        self.server_url = "http://localhost:3001/sse"
        self.openai = AsyncOpenAI(api_key="{GPT_KEY}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        print("Connecting to server...")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python3" if is_python else "node"
        print(f"Using command: {command}")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        print("Server parameters:", server_params)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        print(f"Session created...")

        await self.session.initialize()
        print("Session initialized...")

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def connect_to_server_sse(self):
        print(f"Connecting to SSE server at {self.server_url}...")

        async with sse_client(url=self.server_url) as streams:
            # Create the client session with the streams
            async with ClientSession(*streams) as session:
                # Initialize the session
                await session.initialize()

                # List available tools
                response = await session.list_tools()
                print("Available tools:", [tool.name for tool in response.tools])

                # Call the greet tool
                result = await session.call_tool("greet", {"name": "Bob"})
                print("Greeting result:", result.content)

                # Call the add tool
                result = await session.call_tool("add", {"a": 10, "b": 32})
                print("Addition result:", result.content)

    async def get_openai_functions(self):
        """Convert MCP tools to OpenAI function-calling format."""
        tool_list = await self.session.list_tools()
        functions = []
        for tool in tool_list.tools:
            fn_schema = {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "required": []
            }
            if tool.inputSchema and tool.inputSchema.get("type") == "object":
                fn_schema["parameters"] = tool.inputSchema
            functions.append(fn_schema)
        print(functions)
        return functions

    async def process_user_message(self, user_message: str) -> str:
        if not self.session:
            raise RuntimeError("No MCP session found. Did you call connect_to_server?")

        functions = await self.get_openai_functions()
        print(functions)
        # First call to GPT: provide user's message + available functions
        response = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_message}],
            functions=functions,
            function_call="auto"
        )
        assistant_message = response.choices[0].message
        print(f"\n[Assistant] {assistant_message}")

        if hasattr(assistant_message, "function_call") and assistant_message.function_call:
            fn_name = assistant_message.function_call.name
            fn_args_json = assistant_message.function_call.arguments
            try:
                fn_args = json.loads(fn_args_json) if fn_args_json else {}
            except json.JSONDecodeError:
                return "[Error] GPT produced invalid JSON arguments."

            print(f"\n[Tool request] GPT wants to call {fn_name} with {fn_args}")
            tool_result = await self.session.call_tool(fn_name, fn_args)

            # Second call to GPT: provide the tool's result
            followup = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_message},
                    assistant_message.to_dict(),
                    {"role": "function", "name": fn_name, "content": tool_result.content}
                ]
            )
            print(f"\n[Followup] {followup}")
            return followup.choices[0].message.content
        else:
            # GPT didn't request a function
            return assistant_message.content

async def main():
    client = MCPClient()
    server_script_path = "server.py"  # Change this to your server script path
    try:
        # await client.connect_to_server(server_script_path)
        await client.connect_to_server_sse()
        # await client.get_openai_functions()
        # await client.process_user_message("What is 2 + 3?")
    finally:
        await client.exit_stack.aclose()
        print("Exit stack closed.")

    # Example usage of the client session
    # You can call tools or resources here using client.session

if __name__ == "__main__":
    import sys
    asyncio.run(main())