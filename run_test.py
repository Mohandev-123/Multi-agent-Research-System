"""Quick runner script that bypasses interactive input."""
from pipeline import run_research_pipeline

state = run_research_pipeline("The Impact of War on the Stock Market")
print("\n\n=== PIPELINE COMPLETE ===")
print(f"Sources found: {len(state.search_results)}")
print(f"Sources scraped: {sum(1 for s in state.sources if s.scrape_success)}/{len(state.sources)}")
print(f"Revisions: {state.revision_count}")
if state.critic_feedback:
    print(f"Final score: {state.critic_feedback.score}/10")
    print(f"Decision: {state.critic_feedback.decision}")
if state.warnings:
    print("Warnings:")
    for w in state.warnings:
        print(f"  - {w}")
