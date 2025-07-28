#!/usr/bin/env python3
"""
Test script for LLM providers to verify they are responding correctly.
This script tests OpenAI, Groq, and Ollama providers.
"""

import os
import sys
import time
from typing import Optional, Dict, Any
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the path to import llm_generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_generator import LLMAgentFactory, LLMProvider, BaseLLMAgent


class TestLLMProviders(unittest.TestCase):
    """Test class for LLM providers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_prompt = "What is 2+2? Please respond with just the number."
        self.expected_keywords = ["4", "four"]
        
    def test_openai_availability(self):
        """Test if OpenAI agent can be created and checks availability"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4.1-nano"
            )
            self.assertIsInstance(agent, BaseLLMAgent)
            
            # Test availability check (should work even without valid API key)
            availability = agent.is_available()
            self.assertIsInstance(availability, bool)
            print(f"OpenAI availability: {availability}")
            
        except Exception as e:
            self.fail(f"Failed to create OpenAI agent: {e}")
    
    def test_groq_availability(self):
        """Test if Groq agent can be created and checks availability"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.GROQ,
                model_name="llama-3.1-8b-instant"
            )
            self.assertIsInstance(agent, BaseLLMAgent)
            
            # Test availability check
            availability = agent.is_available()
            self.assertIsInstance(availability, bool)
            print(f"Groq availability: {availability}")
            
        except Exception as e:
            self.fail(f"Failed to create Groq agent: {e}")
    
    def test_ollama_availability(self):
        """Test if Ollama agent can be created and checks availability"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.OLLAMA,
                model_name="qwen3:0.6b"
            )
            self.assertIsInstance(agent, BaseLLMAgent)
            
            # Test availability check
            availability = agent.is_available()
            self.assertIsInstance(availability, bool)
            print(f"Ollama availability: {availability}")
            
        except Exception as e:
            self.fail(f"Failed to create Ollama agent: {e}")
    
    def test_openai_llm_creation(self):
        """Test OpenAI LLM creation"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4.1-nano"
            )
            
            if agent.is_available():
                llm = agent.create_llm()
                self.assertIsNotNone(llm)
                print(f"OpenAI LLM created successfully: {type(llm).__name__}")
            else:
                print("OpenAI not available - skipping LLM creation test")
                
        except ValueError as e:
            if "API_KEY not found" in str(e):
                print("OpenAI API key not found - skipping LLM creation test")
            else:
                raise
        except Exception as e:
            self.fail(f"Failed to create OpenAI LLM: {e}")
    
    def test_groq_llm_creation(self):
        """Test Groq LLM creation"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.GROQ,
                model_name="llama-3.1-8b-instant"
            )
            
            if agent.is_available():
                llm = agent.create_llm()
                self.assertIsNotNone(llm)
                print(f"Groq LLM created successfully: {type(llm).__name__}")
            else:
                print("Groq not available - skipping LLM creation test")
                
        except ValueError as e:
            if "API_KEY not found" in str(e):
                print("Groq API key not found - skipping LLM creation test")
            else:
                raise
        except Exception as e:
            self.fail(f"Failed to create Groq LLM: {e}")
    
    def test_ollama_llm_creation(self):
        """Test Ollama LLM creation"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=LLMProvider.OLLAMA,
                model_name="qwen3:0.6b"
            )
            
            if agent.is_available():
                llm = agent.create_llm()
                self.assertIsNotNone(llm)
                print(f"Ollama LLM created successfully: {type(llm).__name__}")
            else:
                print("Ollama not available - skipping LLM creation test")
                
        except Exception as e:
            self.fail(f"Failed to create Ollama LLM: {e}")


class TestLLMResponses(unittest.TestCase):
    """Test class for actual LLM responses"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_prompt = "What is the capital of France? Please respond with just the city name."
        self.expected_answer = "paris"
        self.simple_math_prompt = "What is 2+2? Respond with just the number."
        
    def _test_llm_response(self, provider: LLMProvider, model_name: str):
        """Helper method to test LLM responses"""
        try:
            agent = LLMAgentFactory.create_agent(
                provider=provider,
                model_name=model_name
            )
            
            if not agent.is_available():
                print(f"{provider.value} not available - skipping response test")
                return
                
            llm = agent.create_llm()
            
            # Test simple question
            start_time = time.time()
            print(f"Testing {provider.value} response for prompt: {self.test_prompt}")
            response = llm.invoke(self.test_prompt)
            if type(response) is not str:
                response = response.content if hasattr(response, 'content') else str(response)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response.strip()), 0)
            
            print(f"{provider.value} Response: {response}")
            print(f"{provider.value} Response time: {response_time:.2f}s")
            
            # Check if response contains expected answer
            if self.expected_answer.lower() in response.lower():
                print(f"✅ {provider.value} provided correct answer")
            else:
                print(f"⚠️  {provider.value} response may be incorrect")
                
            return True
            
        except ValueError as e:
            if "API_KEY not found" in str(e):
                print(f"{provider.value} API key not found - skipping response test")
                return False
            else:
                raise
        except Exception as e:
            print(f"❌ {provider.value} response test failed: {e}")
            return False
    
    def test_openai_response(self):
        """Test OpenAI response"""
        success = self._test_llm_response(LLMProvider.OPENAI, "gpt-4.1-nano")
        if success:
            print("✅ OpenAI response test completed")
    
    def test_groq_response(self):
        """Test Groq response"""
        success = self._test_llm_response(LLMProvider.GROQ, "llama-3.1-8b-instant")
        if success:
            print("✅ Groq response test completed")
    
    def test_ollama_response(self):
        """Test Ollama response"""
        success = self._test_llm_response(LLMProvider.OLLAMA, "qwen3:0.6b")
        if success:
            print("✅ Ollama response test completed")


def run_quick_test():
    """Run a quick test of all providers"""
    print("🚀 Running Quick LLM Provider Test\n")
    print("=" * 50)
    
    providers = [
        (LLMProvider.OPENAI, "gpt-4.1-nano"),
        (LLMProvider.GROQ, "llama-3.1-8b-instant"),
        (LLMProvider.OLLAMA, "qwen3:0.6b")
    ]
    
    results = {}
    
    for provider, model in providers:
        print(f"\n🔍 Testing {provider.value} ({model})...")
        try:
            agent = LLMAgentFactory.create_agent(
                provider=provider,
                model_name=model
            )
            
            # Test availability
            available = agent.is_available()
            print(f"   Availability: {'✅ Available' if available else '❌ Not Available'}")
            
            if available:
                # Test LLM creation
                llm = agent.create_llm()
                print(f"   LLM Creation: ✅ Success ({type(llm).__name__})")
                
                # Test simple response
                try:
                    response = llm.invoke("Say hello")
                    if type(response) is not str:
                        response = response.content if hasattr(response, 'content') else str(response)
                    if response and len(response.strip()) > 0:
                        print(f"   Response Test: ✅ Success")
                        print(f"   Sample Response: {response[:100]}...")
                        results[provider.value] = "✅ Full Success"
                    else:
                        print(f"   Response Test: ❌ Empty response")
                        results[provider.value] = "⚠️ Partial Success"
                except Exception as e:
                    print(f"   Response Test: ❌ Failed ({str(e)[:50]}...)")
                    results[provider.value] = "⚠️ Creation Only"
            else:
                results[provider.value] = "❌ Not Available"
                
        except Exception as e:
            print(f"   Error: ❌ {str(e)[:50]}...")
            results[provider.value] = "❌ Failed"
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    for provider, result in results.items():
        print(f"   {provider.upper():<10} {result}")
    
    print("\n💡 Tips:")
    print("   - Ensure API keys are set in .env file")
    print("   - For Ollama: make sure service is running locally")
    print("   - Check internet connection for cloud providers")


def run_individual_quick_test(providers: list, custom_model: str = None):
    """Run quick test for specific providers"""
    print(f"🚀 Running Quick Test for: {', '.join(providers).upper()}\n")
    print("=" * 50)
    
    provider_configs = {
        "openai": (LLMProvider.OPENAI, custom_model or "gpt-4.1-nano"),
        "groq": (LLMProvider.GROQ, custom_model or "llama-3.1-8b-instant"),
        "ollama": (LLMProvider.OLLAMA, custom_model or "qwen3:0.6b")
    }
    
    results = {}
    
    for provider_name in providers:
        if provider_name in provider_configs:
            provider, model = provider_configs[provider_name]
            print(f"\n🔍 Testing {provider.value} ({model})...")
            try:
                agent = LLMAgentFactory.create_agent(
                    provider=provider,
                    model_name=model
                )
                
                # Test availability
                available = agent.is_available()
                print(f"   Availability: {'✅ Available' if available else '❌ Not Available'}")
                
                if available:
                    # Test LLM creation
                    llm = agent.create_llm()
                    print(f"   LLM Creation: ✅ Success ({type(llm).__name__})")
                    
                    # Test simple response
                    try:
                        response = llm.invoke("Say hello")
                        if type(response) is not str:
                            response = response.content if hasattr(response, 'content') else str(response)
                        if response and len(response.strip()) > 0:
                            print(f"   Response Test: ✅ Success")
                            print(f"   Sample Response: {response[:100]}...")
                            results[provider.value] = "✅ Full Success"
                        else:
                            print(f"   Response Test: ❌ Empty response")
                            results[provider.value] = "⚠️ Partial Success"
                    except Exception as e:
                        print(f"   Response Test: ❌ Failed ({str(e)[:50]}...)")
                        results[provider.value] = "⚠️ Creation Only"
                else:
                    results[provider.value] = "❌ Not Available"
                    
            except Exception as e:
                print(f"   Error: ❌ {str(e)[:50]}...")
                results[provider.value] = "❌ Failed"
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    for provider, result in results.items():
        print(f"   {provider.upper():<10} {result}")


def run_specific_tests(providers: list, args):
    """Run specific types of tests"""
    print(f"🎯 Running Specific Tests for: {', '.join(providers).upper()}\n")
    
    provider_configs = {
        "openai": (LLMProvider.OPENAI, args.model or "gpt-4.1-nano"),
        "groq": (LLMProvider.GROQ, args.model or "llama-3.1-8b-instant"),
        "ollama": (LLMProvider.OLLAMA, args.model or "qwen3:0.6b")
    }
    
    for provider_name in providers:
        if provider_name in provider_configs:
            provider, model = provider_configs[provider_name]
            print(f"\n🔍 Testing {provider.value} ({model})...")
            
            try:
                agent = LLMAgentFactory.create_agent(
                    provider=provider,
                    model_name=model
                )
                
                if args.availability:
                    available = agent.is_available()
                    print(f"   ✅ Availability: {'Available' if available else 'Not Available'}")
                
                if args.creation and (not args.availability or agent.is_available()):
                    try:
                        llm = agent.create_llm()
                        print(f"   ✅ LLM Creation: Success ({type(llm).__name__})")
                    except Exception as e:
                        print(f"   ❌ LLM Creation: Failed ({str(e)[:50]}...)")
                
                if args.response and agent.is_available():
                    try:
                        llm = agent.create_llm()
                        response = llm.invoke("What is 2+2?")
                        if type(response) is not str:
                            response = response.content if hasattr(response, 'content') else str(response)
                        if response and len(response.strip()) > 0:
                            print(f"   ✅ Response Test: Success")
                            print(f"   📝 Response: {response[:100]}...")
                        else:
                            print(f"   ❌ Response Test: Empty response")
                    except Exception as e:
                        print(f"   ❌ Response Test: Failed ({str(e)[:50]}...)")
                        
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:50]}...")


def run_individual_unit_tests(providers: list):
    """Run unit tests for specific providers"""
    print(f"🧪 Running Unit Tests for: {', '.join(providers).upper()}")
    
    suite = unittest.TestSuite()
    
    # Add specific tests based on providers
    if "openai" in providers:
        suite.addTest(TestLLMProviders('test_openai_availability'))
        suite.addTest(TestLLMProviders('test_openai_llm_creation'))
    
    if "groq" in providers:
        suite.addTest(TestLLMProviders('test_groq_availability'))
        suite.addTest(TestLLMProviders('test_groq_llm_creation'))
    
    if "ollama" in providers:
        suite.addTest(TestLLMProviders('test_ollama_availability'))
        suite.addTest(TestLLMProviders('test_ollama_llm_creation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def run_individual_response_tests(providers: list, custom_model: str = None):
    """Run response tests for specific providers"""
    print(f"💬 Running Response Tests for: {', '.join(providers).upper()}")
    
    suite = unittest.TestSuite()
    
    # Create a custom test class for individual providers
    class IndividualTestLLMResponses(TestLLMResponses):
        def test_individual_responses(self):
            provider_configs = {
                "openai": (LLMProvider.OPENAI, custom_model or "gpt-4.1-nano"),
                "groq": (LLMProvider.GROQ, custom_model or "llama-3.1-8b-instant"),
                "ollama": (LLMProvider.OLLAMA, custom_model or "qwen3:0.6b")
            }
            
            for provider_name in providers:
                if provider_name in provider_configs:
                    provider, model = provider_configs[provider_name]
                    print(f"\n🔍 Testing {provider.value} responses...")
                    success = self._test_llm_response(provider, model)
                    if success:
                        print(f"✅ {provider.value} response test completed")
    
    suite.addTest(IndividualTestLLMResponses('test_individual_responses'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def print_usage_examples():
    """Print usage examples for the test script"""
    print("📚 Usage Examples:")
    print("=" * 50)
    print("# Test all providers (default)")
    print("python test_llm_provider.py")
    print()
    print("# Test only OpenAI")
    print("python test_llm_provider.py --openai")
    print()
    print("# Test only Groq with custom model")
    print("python test_llm_provider.py --groq --model llama-3.1-70b-versatile")
    print()
    print("# Test only availability for all providers")
    print("python test_llm_provider.py --availability")
    print()
    print("# Test only OpenAI availability and creation")
    print("python test_llm_provider.py --openai --availability --creation")
    print()
    print("# Test responses for Ollama and Groq")
    print("python test_llm_provider.py --ollama --groq --responses")
    print()
    print("# Run unit tests for OpenAI only")
    print("python test_llm_provider.py --openai --unittest")
    print()
    print("# Run all tests for specific provider")
    print("python test_llm_provider.py --groq --all")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_llm_provider.py                          # Test all providers
  python test_llm_provider.py --openai                 # Test only OpenAI
  python test_llm_provider.py --groq --model llama-3.1-70b-versatile  # Test Groq with custom model
  python test_llm_provider.py --availability           # Test only availability
  python test_llm_provider.py --openai --creation      # Test OpenAI creation only
  python test_llm_provider.py --examples               # Show detailed examples
        """
    )
    
    # General test modes
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--unittest", action="store_true", help="Run unit tests")
    parser.add_argument("--responses", action="store_true", help="Test actual responses")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # Individual provider testing
    parser.add_argument("--openai", action="store_true", help="Test only OpenAI")
    parser.add_argument("--groq", action="store_true", help="Test only Groq")
    parser.add_argument("--ollama", action="store_true", help="Test only Ollama")
    
    # Specific test types
    parser.add_argument("--availability", action="store_true", help="Test only availability")
    parser.add_argument("--creation", action="store_true", help="Test only LLM creation")
    parser.add_argument("--response", action="store_true", help="Test only responses")
    
    # Model specification
    parser.add_argument("--model", type=str, help="Specify model name for testing")
    
    # Help and examples
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    # Show examples if requested
    if args.examples:
        print_usage_examples()
        exit(0)
    
    # Determine which providers to test
    test_providers = []
    if args.openai:
        test_providers.append("openai")
    if args.groq:
        test_providers.append("groq")
    if args.ollama:
        test_providers.append("ollama")
    
    # If no individual providers specified, test all
    if not test_providers:
        test_providers = ["openai", "groq", "ollama"]
    
    # Run specific test types if requested
    if any([args.availability, args.creation, args.response]):
        run_specific_tests(test_providers, args)
    elif args.quick or (not any([args.unittest, args.responses, args.all])):
        if len(test_providers) < 3:
            run_individual_quick_test(test_providers, args.model)
        else:
            run_quick_test()
    
    if args.unittest or args.all:
        print("\n🧪 Running Unit Tests...")
        if len(test_providers) < 3:
            run_individual_unit_tests(test_providers)
        else:
            unittest.main(argv=[''], verbosity=2, exit=False)
    
    if args.responses or args.all:
        print("\n💬 Running Response Tests...")
        if len(test_providers) < 3:
            run_individual_response_tests(test_providers, args.model)
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(TestLLMResponses)
            unittest.TextTestRunner(verbosity=2).run(suite)
