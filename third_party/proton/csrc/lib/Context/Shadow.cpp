#include "Context/Shadow.h"

#include <stdexcept>
#include <thread>

namespace proton {

void ShadowContextSource::enterScope(const Scope &scope) {
  if (!mainContextStack) {
    mainContextStack = &threadContextStack;
    contextInitialized = true;
  }
  if (!contextInitialized && mainContextStack != &threadContextStack) {
    threadContextStack = *mainContextStack;
    contextInitialized = true;
  }
  threadContextStack.push_back(scope);
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (threadContextStack.empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (threadContextStack.back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  threadContextStack.pop_back();
}

/*static*/ thread_local std::vector<Context>
    ShadowContextSource::threadContextStack;

/*static*/ thread_local bool ShadowContextSource::contextInitialized = false;

} // namespace proton
