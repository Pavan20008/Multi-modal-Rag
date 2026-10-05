"""
CLI entry point.

Usage:
    python main.py ingest --pdf path/to/document.pdf
    python main.py query --q "What does the architecture diagram show?"
"""
import argparse

from pipeline import ingest, answer_query


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Process a PDF into the vector store")
    ingest_parser.add_argument("--pdf", required=True, help="Path to the PDF file")

    query_parser = subparsers.add_parser("query", help="Ask a question about the ingested PDF")
    query_parser.add_argument("--q", required=True, help="Your question")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf)

    elif args.command == "query":
        result = answer_query(args.q)
        print("\n=== ANSWER ===")
        print(result["answer"])

        print("\n=== SOURCES ===")
        for c in result["text_chunks"]:
            print(f"- Page {c['page']} (text)")
        for i in result["images"]:
            print(f"- Page {i['page']} (image: {i['path']})")


if __name__ == "__main__":
    main()