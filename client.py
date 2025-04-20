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
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.deepseek = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.remote_session = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
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
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def connect_to_remote_server(self, server_url: str, api_key: str = None):
        """Connect to a remote MCP server using Server-Sent Events (SSE)
        
        Args:
            server_url: Full URL of the remote MCP server
            api_key: Optional API key for authentication
        """
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

    async def process_query(self, query: str) -> str:
        """Process a query using DeepSeek and available tools"""
        # List available tools first
        response = await self.session.list_tools()
        tools = response.tools

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Convert available tools to tool_choice format
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tools]

        # Initial DeepSeek API call
        response = self.deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=available_tools,
            max_tokens=1000,
            stream=False
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        # Extract text and tool calls from the response
        assistant_response = response.choices[0].message.content or ""
        assistant_tool_calls = response.choices[0].message.tool_calls or []

        if assistant_response:
            print ("initial response: ", assistant_response)
            final_text.append(assistant_response)
            messages.append({
                "role": "assistant",
                "content": assistant_response
            })

        # Process tool calls
        if assistant_tool_calls:
            # Prepare messages to include tool calls and their results
            messages.append({
                "role": "assistant",
                "tool_calls": assistant_tool_calls
            })

            for tool_call in assistant_tool_calls:
                # Process tool calls
                tool_name = tool_call.function.name
                
                # Parse arguments from JSON string to dictionary
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    # Fallback to original arguments if parsing fails
                    tool_args = tool_call.function.arguments

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)

                # Extract text content if it's a TextContent object
                result_text = result.content.text if hasattr(result.content, 'text') else str(result.content)
                
                # Prepare tool result message
                tool_result_message = f"Tool {tool_name} executed with result: {result.content}"
                print ("tool response: ", result_text)
                # Add tool processing logic
                tool_result = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result_message
                }
                tool_results.append(tool_result)

                # Add tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })

        # If there are tool results, make a follow-up call
        if tool_results:
            response = self.deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=1000,
                stream=False
            )
            
            # Append final response text
            final_response = response.choices[0].message.content or ""
            if final_response:
                
                final_text.append(final_response)

        return " ".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
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
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
