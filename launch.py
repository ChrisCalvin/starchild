"""
SASAIE System Launcher

Main entry point for the SASAIE system launcher.
Provides a CLI for managing the SASAIE trading infrastructure.
"""

import asyncio
import logging
import sys
import time

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from sasaie_trader.launcher.orchestrator import SASAIEOrchestrator

# --- Configuration ---
load_dotenv(dotenv_path="configs/launcher/.env")
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) output.")
@click.pass_context
def cli(ctx, verbose):
    """
    SASAIE System Launcher CLI.

    Orchestrates the SASAIE system, including the containerized Hummingbot instance.
    """
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option("--profile", "-p", default="development", help="Launch profile name to use.")
@click.option("--force-restart", is_flag=True, help="Force restart if services are already running.")
@click.option("--password", "-P", help="Hummingbot password (will prompt if not provided).", hide_input=True, envvar='HUMMINGBOT_PASSWORD')
@click.pass_context
def start(ctx, profile, force_restart, password):
    """
    Starts the SASAIE system, including the Hummingbot container.
    """
    console.print(Panel(f"[bold blue]🚀 Starting SASAIE System[/bold blue]\nProfile: [cyan]{profile}[/cyan]", border_style="blue"))
    
    orchestrator = SASAIEOrchestrator(profile_name=profile)
    try:
        success = asyncio.run(orchestrator.launch_system(force_restart=force_restart, hummingbot_password=password))
        if success:
            console.print(Panel("[bold green]✅ System Started Successfully[/bold green]", border_style="green"))
            console.print("The system is running in the background. Use 'status' to check on services or 'stop' to shut down.")
        else:
            console.print(Panel("[bold red]❌ System Start Failed[/bold red]", border_style="red"))
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown signal received. Stopping system...[/yellow]")
        asyncio.run(orchestrator.stop_system())
        console.print(Panel("[bold green]✅ System Stopped Successfully[/bold green]"))

@cli.command()
@click.option("--profile", "-p", default="development", help="Launch profile name to use.")
@click.pass_context
def stop(ctx, profile):
    """
    Stops the SASAIE system and its services.
    """
    console.print(Panel("[bold yellow]🛑 Stopping SASAIE System[/bold yellow]", border_style="yellow"))
    orchestrator = SASAIEOrchestrator(profile_name=profile)
    success = asyncio.run(orchestrator.stop_system())
    if success:
        console.print(Panel("[bold green]✅ System Stopped Successfully[/bold green]", border_style="green"))
    else:
        console.print(Panel("[bold red]❌ System Stop Failed[/bold red]", border_style="red")),
        sys.exit(1)

@cli.command()
@click.option("--profile", "-p", default="development", help="Launch profile name to use.") # Add this line
@click.pass_context
def status(ctx, profile): # Add profile to function signature
    """
    Checks the status of the SASAIE system services.
    """
    console.print(Panel("[bold cyan]📊 Checking System Status[/bold cyan]", border_style="cyan"))
    orchestrator = SASAIEOrchestrator(profile_name=profile) # Use the passed profile
    status = asyncio.run(orchestrator.get_system_status())

    if not status or status.get("status") == "error":
        console.print(Panel(f"[bold red]❌ Could not retrieve status.[/bold red]\nError: {status.get('error')}", border_style="red")),
        sys.exit(1)

    table = Table(title=f"SASAIE System Status: [bold { 'green' if status.get('status') == 'running' else 'red' }]{status.get('status', 'UNKNOWN').upper()}[/]")
    table.add_column("Service", style="cyan")
    table.add_column("Container ID", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Health", style="green")

    services = status.get("services", {})
    if not services:
        console.print("No running services found.")
    else:
        for name, info in services.items():
            health = info.get('health', 'unknown')
            health_color = "green" if health == "healthy" else "red" if health == "unhealthy" else "yellow"
            table.add_row(name, info.get('container_id', 'N/A')[:12], info.get('status', 'N/A'), f"[{health_color}]{health}[/]")
        console.print(table)

    hummingbot_strategy = status.get('hummingbot_strategy')
    if hummingbot_strategy:
        strategy_table = Table(title="Hummingbot Strategy Status")
        strategy_table.add_column("Parameter", style="cyan")
        strategy_table.add_column("Value", style="magenta")
        strategy_table.add_row("Running", str(hummingbot_strategy.get('running')))
        strategy_table.add_row("Strategy", hummingbot_strategy.get('strategy'))
        strategy_table.add_row("Exchange", hummingbot_strategy.get('exchange'))
        strategy_table.add_row("Market", hummingbot_strategy.get('market'))
        if 'error' in hummingbot_strategy:
            strategy_table.add_row("Error", hummingbot_strategy.get('error'))
        console.print(strategy_table)

if __name__ == "__main__":
    cli()
