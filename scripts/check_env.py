#!/usr/bin/env python3
"""Check environment setup for L2L simulation."""
import os
import sys

def check_env():
    """Check if required environment variables are set."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    issues = []

    # Check OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY: Set ({len(openai_key)} chars)")
    else:
        print("✗ OPENAI_API_KEY: Not set")
        issues.append("OPENAI_API_KEY")

    # Check Anthropic API key (support both standard and legacy names)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_key")
    if anthropic_key:
        key_name = "ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else "anthropic_key"
        print(f"✓ {key_name}: Set ({len(anthropic_key)} chars)")
    else:
        print("✗ ANTHROPIC_API_KEY: Not set")
        issues.append("ANTHROPIC_API_KEY")

    # Check Gemini API key (support both standard and legacy names)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_key")
    if gemini_key:
        key_name = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "gemini_key"
        print(f"✓ {key_name}: Set ({len(gemini_key)} chars)")
    else:
        print("✗ GEMINI_API_KEY: Not set")
        issues.append("GEMINI_API_KEY")

    print("=" * 60)

    if issues:
        print("\n⚠️  Missing API keys:")
        for key in issues:
            print(f"   export {key}='your-key-here'")
        print("\nOr add to .env file:")
        for key in issues:
            print(f"   {key}=your-key-here")
        return False
    else:
        print("✓ All API keys are set!")
        return True

if __name__ == "__main__":
    success = check_env()
    sys.exit(0 if success else 1)
