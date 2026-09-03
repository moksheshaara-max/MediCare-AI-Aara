"""
MediCare AI - PubMed Live Research Search Module
Fetches real-time, peer-reviewed medical research papers from NIH PubMed.
Features custom User-Agent, API key support, and Progressive Query Broadening.
"""

import os
import re
import time
import requests
import xmltodict
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "").strip()
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

MAX_RESULTS = 5
REQUEST_TIMEOUT = 12

HEADERS = {
    "User-Agent": "MediCareAI/1.0 (VIT Chennai B.Tech Project; medicare_ai@vit.ac.in)",
    "Accept": "*/*"
}


def clean_query_terms(query: str) -> List[str]:
    """
    Extract key medical terms from user prompts for PubMed API.
    Returns a list of medical terms sorted by importance.
    """
    clean = re.sub(r'[^\w\s\-/]', ' ', query)
    words = clean.split()
    
    stop_words = {
        "what", "are", "the", "latest", "clinical", "trial", "evidence", "regarding",
        "efficacy", "and", "safety", "versus", "selective", "in", "patients", "with",
        "co", "existing", "specifically", "address", "based", "on", "recent", "literature",
        "do", "trials", "demonstrate", "without", "worsening", "of", "against", "how",
        "these", "novel", "supersede", "traditional", "management", "report", "between",
        "vs", "single", "evaluate", "coexisting", "please", "analyze", "explain",
        "according", "to", "for", "from", "this", "that", "which", "when", "where",
        "who", "whom", "whose", "why", "does", "did", "will", "would", "could",
        "should", "can", "may", "might", "must", "have", "has", "had", "been",
        "being", "is", "was", "were", "be", "am", "a", "an", "as", "at", "by",
        "into", "through", "during", "before", "after", "above", "below", "up",
        "down", "out", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "about", "into", "like", "using", "used",
        "include", "including", "provide", "gives", "give", "get", "tell", "show",
        "describe", "compare", "difference", "differences", "both", "either", "neither"
    }
    
    return [w for w in words if w.lower() not in stop_words and len(w) > 2]


def search_pubmed_ids(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """
    Search PubMed for article IDs using Progressive Query Broadening.
    Tries 3 terms -> 2 terms -> 1 term until papers are found.
    """
    terms = clean_query_terms(query)
    
    if not terms:
        terms = query.split()[:3]
    
    # Build progressive search attempts from specific to broad
    search_attempts = []
    if len(terms) >= 3:
        search_attempts.append(" ".join(terms[:3]))
    if len(terms) >= 2:
        search_attempts.append(" ".join(terms[:2]))
    if len(terms) >= 1:
        search_attempts.append(terms[0])
    
    url = f"{BASE_URL}/esearch.fcgi"
    
    for term in search_attempts:
        print(f"    PubMed Searching: '{term}'...")
        
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "tool": "medicare_ai",
            "email": "medicare_ai@vit.ac.in"
        }
        
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY
        
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            if id_list:
                print(f"    ✓ PubMed Found {len(id_list)} papers for '{term}'!")
                return id_list
                
        except Exception as e:
            print(f"    PubMed Search Error for '{term}': {e}")
            continue
    
    return []


def fetch_pubmed_papers(ids: List[str]) -> List[Dict]:
    """Fetch abstracts and metadata for given PubMed IDs."""
    if not ids:
        return []
    
    url = f"{BASE_URL}/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "xml",
        "tool": "medicare_ai",
        "email": "medicare_ai@vit.ac.in"
    }
    
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    
    for attempt in range(3):
        try:
            time.sleep(0.3)
            response = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = xmltodict.parse(response.content)
            articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
            
            if isinstance(articles, dict):
                articles = [articles]
            
            papers = []
            for article in articles:
                paper = parse_article(article)
                if paper:
                    papers.append(paper)
            return papers
            
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            print(f"PubMed Fetch Error: {e}")
            return []
    
    return []


def parse_article(article: Dict) -> Dict:
    """Extract structured details from a PubMed article XML dictionary."""
    try:
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})
        
        # PMID
        pmid = medline.get("PMID", {})
        if isinstance(pmid, dict):
            pmid = pmid.get("#text", "N/A")
        
        # Title
        title = article_data.get("ArticleTitle", "No Title")
        if isinstance(title, dict):
            title = title.get("#text", str(title))
        
        # Abstract
        abstract_data = article_data.get("Abstract", {})
        abstract_text = abstract_data.get("AbstractText", "")
        
        if isinstance(abstract_text, list):
            parts = []
            for sec in abstract_text:
                if isinstance(sec, dict):
                    lbl = sec.get("@Label", "")
                    txt = sec.get("#text", "")
                    parts.append(f"{lbl}: {txt}" if lbl else txt)
                else:
                    parts.append(str(sec))
            abstract_text = " ".join(parts)
        elif isinstance(abstract_text, dict):
            abstract_text = abstract_text.get("#text", "")
        
        # Authors
        author_list = article_data.get("AuthorList", {}).get("Author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        
        authors = []
        for author in author_list[:3]:
            last_name = author.get("LastName", "")
            initials = author.get("Initials", "")
            if last_name:
                authors.append(f"{last_name} {initials}".strip())
        
        author_str = ", ".join(authors)
        if len(author_list) > 3:
            author_str += ", et al."
        
        # Journal & Year
        journal = article_data.get("Journal", {})
        journal_title = journal.get("Title", "Medical Journal")
        pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "Recent")
        
        return {
            "pmid": str(pmid),
            "title": str(title),
            "abstract": str(abstract_text) if abstract_text else "Abstract unavailable.",
            "authors": author_str or "Unknown authors",
            "journal": str(journal_title),
            "year": str(year),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
        
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def search_pubmed_live(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """Main search wrapper."""
    ids = search_pubmed_ids(query, max_results)
    if not ids:
        return []
    return fetch_pubmed_papers(ids)


def format_pubmed_for_prompt(papers: List[Dict]) -> str:
    """Format papers as context string for Gemini."""
    if not papers:
        return ""
    
    formatted = []
    for i, p in enumerate(papers, 1):
        formatted.append(
            f"[Research Paper {i} | PMID: {p['pmid']}]\n"
            f"Title: {p['title']}\n"
            f"Authors: {p.get('authors', 'N/A')}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"Abstract: {p['abstract'][:700]}\n"
        )
    return "\n---\n".join(formatted)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PubMed Integration with Progressive Query Broadening")
    print("=" * 60)
    
    test_query = "What is the clinical trial evidence for Tirzepatide versus Semaglutide in NASH and diabetic kidney disease?"
    print(f"Query: {test_query}\n")
    
    results = search_pubmed_live(test_query, max_results=3)
    
    if results:
        print(f"\n✓ SUCCESS: Retrieved {len(results)} research papers!\n")
        for i, paper in enumerate(results, 1):
            print(f"[{i}] {paper['title'][:100]}")
            print(f"    Journal: {paper['journal']} ({paper['year']}) | PMID: {paper['pmid']}\n")