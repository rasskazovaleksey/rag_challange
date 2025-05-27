# server.py
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn

from rabbit.config_loader import ConfigLoader
from rabbit.sender import PdfSender
import json

from questions import QuestionExtractor


#TODO fix global vars
config = ConfigLoader().get_rabbitmq_config()
sender = PdfSender(config)



# Create an MCP server
mcp = FastMCP("Demo")


@mcp.resource("pdfs://{filename}")
def get_file(filename: str) -> str:
    sender.send_pdf(f"pdfs/{filename}.pdf")
    sender.close()
    return 'File sent to rabbit queue'

@mcp.resource("getsha://{company_name}")
def get_sha(company_name: str) -> str:
    with open('subset.json') as f:
        data = json.load(f)
        print(company_name)
        for item in data:
            if item['company_name'] == company_name:
                file_name = item['sha1']
                return f'File name {file_name}'
            
    return 'Not found'


@mcp.prompt()
def get_prompt(name: str) -> str:
    with open(f'prompt/{name}.txt') as f:
        return f.read()


@mcp.tool()
def question_extract(question: str) -> str:
    extractor = QuestionExtractor()
    extract = extractor.extract(question)
    return extract

@mcp.tool()
def get_synonyms(extract: str) -> str:
    extractor = QuestionExtractor()
    close_metrics = extractor.get_synonyms(extract)
    return close_metrics


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the MCP server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    # Create Starlette app with SSE support
    starlette_app = create_starlette_app(mcp_server = mcp._mcp_server, debug=True)

    port = 3001
    print(f"Starting MCP server with SSE transport on port {port}...")
    print(f"SSE endpoint available at: http://localhost:{port}/sse")

    # Run the server using uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)