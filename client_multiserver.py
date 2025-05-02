import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
import httpx
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
import os
import dotenv
import sys

dotenv.load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.sessions = {}  # Dictionary to store multiple sessions
        self.exit_stack = AsyncExitStack()
        self.deepseek = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.remote_sessions = {}  # Dictionary to store multiple remote sessions
        self.all_tools = {}  # Dictionary to store all available tools
        self.available_tools = []
        self.tool_servers = {}  # Map tool names to their servers
        self.messages = []  # Store conversation history

    async def connect_to_server(self, server_script_path: str, server_id: str = None):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
            server_id: Optional identifier for the server
        """
        if server_id is None:
            server_id = f"local_{len(self.sessions)}"
            
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        # Store the session
        self.sessions[server_id] = {
            'session': session,
            'stdio': stdio,
            'write': write
        }
        
        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server {server_id} with tools:", [tool.name for tool in tools])
        return server_id

    async def connect_to_remote_server(self, server_url: str, server_id: str = None, api_key: str = None):
        """Connect to a remote MCP server using Server-Sent Events (SSE)"""
        if server_id is None:
            server_id = f"remote_{len(self.remote_sessions)}"
            
        # Prepare headers
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Create an async HTTP client
        async with httpx.AsyncClient() as client:
            try:
                # Establish SSE connection
                async with client.stream('GET', server_url, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Initialize remote session
                    self.remote_session = {
                        'url': server_url,
                        'connection': response
                    }
                    
                    print(f"\nConnected to remote MCP server: {server_url}")
                    
                    # List available tools
                    tools_response = await self.list_remote_tools()
                    print("\nAvailable remote tools:", [tool['name'] for tool in tools_response])
                    
                    return self.remote_session

            except httpx.HTTPStatusError as e:
                print(f"Failed to connect to remote server: {e}")
                raise

    async def list_remote_tools(self):
        """List tools available on the remote MCP server"""
        if not self.remote_session:
            raise RuntimeError("Not connected to a remote server")
        
        # Prepare request to list tools
        list_tools_request = {
            "type": "list_tools",
            "request_id": str(uuid.uuid4())
        }
        
        # Send list tools request
        await self.send_remote_request(list_tools_request)
        
        # Wait for and return tools response
        return await self.receive_remote_response()

    async def send_remote_request(self, request):
        """Send a request to the remote MCP server"""
        if not self.remote_session:
            raise RuntimeError("Not connected to a remote server")
        
        # Implement request sending logic
        # This might involve sending the request through SSE or a specific endpoint
        pass

    async def receive_remote_response(self):
        """Receive a response from the remote MCP server"""
        if not self.remote_session:
            raise RuntimeError("Not connected to a remote server")
        
        # Implement response receiving logic
        # Parse SSE events and extract the relevant response
        async for event in self.remote_session['connection'].aiter_text():
            if event.startswith('data:'):
                try:
                    response = json.loads(event[5:].strip())
                    return response
                except json.JSONDecodeError:
                    print(f"Failed to parse response: {event}")
        
        raise RuntimeError("No response received")

    async def call_remote_tool(self, tool_name, tool_args):
        """Call a tool on the remote MCP server"""
        if not self.remote_session:
            raise RuntimeError("Not connected to a remote server")
        
        # Prepare tool call request
        tool_request = {
            "type": "call_tool",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "request_id": str(uuid.uuid4())
        }
        
        # Send tool call request
        await self.send_remote_request(tool_request)
        
        # Wait for and return tool call response
        return await self.receive_remote_response()

    async def list_all_tools(self):
        """List tools from all connected servers"""
        all_tools = {}
        
        # Get tools from local servers
        for server_id, session_data in self.sessions.items():
            response = await session_data['session'].list_tools()
            all_tools[server_id] = response.tools
            
        # Get tools from remote servers
        for server_id in self.remote_sessions.keys():
            tools_response = await self.list_remote_tools(server_id)
            all_tools[server_id] = tools_response
            
        return all_tools

    async def get_tools_ready(self):
        self.all_tools = await self.list_all_tools()
        for server_id, tools in self.all_tools.items():
            for tool in tools:
                tool_name = f"{server_id}_{tool.name}"  # Prefix tool name with server_id
                self.tool_servers[tool_name] = {
                    'server_id': server_id,
                    'original_name': tool.name
                }
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
        print("Available functions:", [tool["function"]["name"] for tool in self.available_tools])

       
    async def process_query(self, query: str) -> str:
        """Process a query using DeepSeek and available tools from all servers"""
        # Add the new user query to conversation history
        self.messages.append({"role": "user", "content": query})

        # Initial DeepSeek API call, with user query and available tools
        response = self.deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=self.available_tools,
            max_tokens=1000,
            stream=False
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        # Extract text and tool calls from the response
        assistant_message = response.choices[0].message
        assistant_response = assistant_message.content or ""
        assistant_tool_calls = assistant_message.tool_calls or []
        print("assistant tool calls: ", assistant_tool_calls)
        print("assistant response: ", assistant_response)

        if assistant_response:
            print("initial response: ", assistant_response)
            final_text.append(assistant_response)
            self.messages.append({
                "role": "assistant",
                "content": assistant_response,
            })

        # The LLM has chosen some tools to be used, let's make tool calls
        if assistant_tool_calls:
            # Prepare messages to include tool calls and their results
            self.messages.append({
                "role": "assistant",
                "tool_calls": assistant_tool_calls
            })
            for tool_call in assistant_tool_calls:
                # Get server info from tool name
                tool_name = tool_call.function.name
                if tool_name not in self.tool_servers:
                    continue
                    
                server_info = self.tool_servers[tool_name]
                server_id = server_info['server_id']
                original_tool_name = server_info['original_name']
                
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = tool_call.function.arguments

                # Execute tool call on appropriate server
                if server_id.startswith('local_'):
                    session = self.sessions[server_id]['session']
                    result = await session.call_tool(original_tool_name, tool_args)
                else:
                    result = await self.call_remote_tool(original_tool_name, tool_args, server_id)

                # Extract text content if it's a TextContent object
                result_text = result.content.text if hasattr(result.content, 'text') else str(result.content)
                # Prepare tool result message
                tool_result_message = f"Tool {tool_name} executed with result: {result.content}"
                # Add tool processing logic
                tool_result = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result_message
                }
                tool_results.append(tool_result)
                # Add tool result message
                # messages.append({
                #     "role": "assistant",
                #     "content": None,
                #     "tool_calls": [tool_call]
                # })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result_text
                })

        # If there are tool results, make a follow-up call
        if tool_results:
            response = self.deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                max_tokens=1000,
                stream=False
            )
            
            # Append final response text
            final_response = response.choices[0].message.content or ""
            if final_response:
                final_text.append(final_response)
                self.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

        return " ".join(final_text)
        
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        await self.get_tools_ready()
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources for all servers"""
        await self.exit_stack.aclose()
        # Close any remote connections
        for session in self.remote_sessions.values():
            if 'connection' in session:
                await session['connection'].aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_multiserver.py <server1_path> [server2_path ...]")
        sys.exit(1)
        
    client = MCPClient()
    try:
        # Connect to all provided servers
        for server_path in sys.argv[1:]:
            await client.connect_to_server(server_path)
        
        # Start chat loop
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
