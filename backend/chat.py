"""
MediCare AI - Interactive Chat Interface
Features: UI, Commands, Emergency Detection, Non-Medical Filtering, and PubMed Live Research.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from retrieval.search import (
    search_medical_chunks, 
    check_emergency, 
    get_emergency_message,
    is_medical_question,
    get_non_medical_response
)
from generation.generator import generate_answer, format_sources

console = Console()

# Session state
chat_history = []
last_tb_sources = []
last_pm_sources = []

BOOK_TITLES = {
    "book1.pdf": "Hutchison's Clinical Methods",
    "book2.pdf": "Manual of Practical Medicine",
    "harrisons.pdf": "Harrison's Principles of Internal Medicine",
    "davidson.pdf": "Davidson's Principles of Medicine",
    "kd_tripathi.pdf": "KD Tripathi Pharmacology",
    "icmr_diabetes.pdf": "ICMR Guidelines (Diabetes)",
    "who_essential_meds.pdf": "WHO Essential Medicines",
    "icmr_antimicrobial.pdf": "ICMR Antimicrobial Guidelines",
    "mohfw_stw.pdf": "MoHFW Standard Treatment Workflows",
    "who_hypertension.pdf": "WHO Hypertension Guidelines",
    "Type2Diabetes.pdf": "Clinical Guidelines for Type 2 Diabetes"
}

def print_welcome():
    welcome_text = """
[bold cyan]MediCare AI[/bold cyan] - Your Medical Knowledge Assistant

Powered by RAG technology using Textbooks, Guidelines, & PubMed.

[dim]Type your medical question below, or use commands:[/dim]
  [yellow]/help[/yellow]    - Show help
  [yellow]/clear[/yellow]   - Clear screen  
  [yellow]/history[/yellow] - Show chat history
  [yellow]/sources[/yellow] - Sources of last answer
  [yellow]/quit[/yellow]    - Exit

[dim red]⚠️  This is for informational purposes only.
    Always consult a healthcare professional.[/dim red]
"""
    console.print(Panel(welcome_text.strip(), border_style="cyan", padding=(1, 2)))

def print_help():
    help_text = """
[bold]Available Commands:[/bold]
  [yellow]/help[/yellow]     - Show this help message
  [yellow]/clear[/yellow]    - Clear the screen
  [yellow]/history[/yellow]  - Show all questions asked
  [yellow]/sources[/yellow]  - Show sources from last answer
  [yellow]/quit[/yellow]     - Exit MediCare AI
"""
    console.print(Panel(help_text.strip(), title="Help", border_style="yellow"))

def show_history():
    if not chat_history:
        console.print("[dim]No questions asked yet.[/dim]")
        return
    console.print("\n[bold cyan]Chat History:[/bold cyan]")
    for i, item in enumerate(chat_history, 1):
        console.print(f"  [{i}] {item['question']} [dim](Mode: {item['mode']})[/dim]")
    console.print()

def show_sources():
    if not last_tb_sources and not last_pm_sources:
        console.print("[dim yellow]No verified sources available for the last question.[/dim yellow]")
        return
    
    if last_tb_sources:
        console.print("\n[bold cyan]📚 VERIFIED KNOWLEDGE BASE SOURCES:[/bold cyan]")
        for src in last_tb_sources:
            b_name = BOOK_TITLES.get(src['book'], src['book'])
            console.print(f"  • [green]{b_name}[/green], Page [yellow]{src['page']}[/yellow] [dim](score: {src['score']:.2f})[/dim]")
            
    if last_pm_sources:
        console.print("\n[bold magenta]🔬 LIVE PUBMED RESEARCH PAPERS:[/bold magenta]")
        for p in last_pm_sources:
            console.print(f"  • [bold]{p['title']}[/bold]")
            console.print(f"    Journal: {p['journal']} ({p['year']}) | PMID: [yellow]{p['pmid']}[/yellow] | Link: {p['url']}")
    console.print()

def process_question(question):
    global last_tb_sources, last_pm_sources
    
    # 1. Non-Medical Check
    if is_medical_question(question) == "non_medical":
        console.print("\n")
        console.print(Panel(get_non_medical_response(), title="[bold yellow]Not a Medical Question[/bold yellow]", border_style="yellow", padding=(1, 2)))
        return
    
    # 2. Emergency Check
    if check_emergency(question):
        console.print(Panel(get_emergency_message(), title="[bold red]⚠️ EMERGENCY DETECTED ⚠️[/bold red]", border_style="red", padding=(1, 2)))
        console.print("\n[dim]I can still try to provide information, but please seek help first.[/dim]\n")
    
    # 3. Vector Search
    with Progress(SpinnerColumn(), TextColumn("[cyan]Searching medical knowledge base...[/cyan]"), transient=True, console=console) as progress:
        progress.add_task("search", total=None)
        chunks = search_medical_chunks(question, top_k=12)
    
    console.print(f"[dim]Found {len(chunks)} relevant chunks[/dim]")
    
    # 4. Generate Answer + PubMed Search
    with Progress(SpinnerColumn(), TextColumn("[cyan]Generating answer with AI & Live Research...[/cyan]"), transient=True, console=console) as progress:
        progress.add_task("generate", total=None)
        answer, mode, pubmed_papers = generate_answer(question, chunks)
    
    mode_color = {
        "books_and_research": "green",
        "books": "green",
        "research": "magenta",
        "hybrid": "yellow",
        "fallback": "magenta",
        "error": "red"
    }.get(mode, "white")
    
    # 5. Display Answer
    console.print("\n")
    console.print(Panel(Markdown(answer), title=f"[bold]MediCare AI Response[/bold] [{mode_color}]({mode})[/{mode_color}]", border_style=mode_color, padding=(1, 2)))
    
    chat_history.append({"question": question, "answer": answer, "mode": mode})
    
    # 6. Save & Display Sources
    last_tb_sources, last_pm_sources = format_sources(chunks, mode, pubmed_papers)
    
    if last_tb_sources or last_pm_sources:
        show_sources()
        console.print("[dim italic]These are the only verified sources. Any citations in the answer above that don't match this list are not from your textbooks.[/dim italic]\n")
    else:
        console.print("\n[dim yellow]No verified sources (Answer from general knowledge)[/dim yellow]\n")

def main():
    print_welcome()
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            if not question: continue
            
            q_lower = question.lower()
            if q_lower in ["/quit", "/exit"]:
                console.print("\n[cyan]Thank you for using MediCare AI. Stay healthy![/cyan]\n")
                break
            elif q_lower == "/help": print_help()
            elif q_lower == "/clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
            elif q_lower == "/history": show_history()
            elif q_lower == "/sources": show_sources()
            else:
                process_question(question)
                
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Type /quit to exit properly.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()