#include <iostream>
#include <chrono>
#include <string>
#include <functional>
#include <vector>
#include <type_traits>

// Define a compile-time toggle
#ifndef GRB_ENABLE_TRACING
#define GRB_ENABLE_TRACING 1  // Set to 1 for this example
#endif

// STEP 1: Define our decoration machinery before including original header
#if GRB_ENABLE_TRACING
// Decorator class definition
template<typename ImplFunc>
class DecoratedFunction {
private:
    ImplFunc impl;
    std::string name;
    
    // Helper to get type names
    template<typename T>
    std::string getTypeName() const {
        std::string type_name = typeid(T).name();
        // Simple demangling for common types
        if (std::is_same_v<T, double>) return "double";
        if (std::is_same_v<T, float>) return "float";
        if (std::is_same_v<T, int>) return "int";
        return type_name;
    }
    
    template<typename... Args>
    void printArgumentTypes(Args&&... args) const {
        std::cout << "[DEBUG] Argument types: ";
        ((std::cout << getTypeName<std::decay_t<Args>>() << " "), ...);
        std::cout << std::endl;
    }
    
public:
    DecoratedFunction(ImplFunc impl_func, std::string func_name) 
        : impl(std::move(impl_func)), name(std::move(func_name)) {}
    
    template<typename... Args>
    auto operator()(Args&&... args) const 
        -> decltype(impl(std::forward<Args>(args)...)) {
        
        std::cout << "[DEBUG] Entering function: " << name << " with " << sizeof...(args) << " arguments" << std::endl;
        printArgumentTypes(args...);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = impl(std::forward<Args>(args)...);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[DEBUG] Exiting function: " << name << " (took " << duration.count() << "μs)" << std::endl;
        
        return result;
    }
};

// Factory function
template<typename Func>
auto make_decorated_function(Func&& func, const std::string& name) {
    return DecoratedFunction<std::decay_t<Func>>(std::forward<Func>(func), name);
}


// Step 2: Create decorated replacements for specific functions
namespace grb {
    // Decorated eWiseApply
    template<typename... Args>
    auto decorated_eWiseApply(Args&&... args) 
        -> decltype(grb::eWiseApply(std::forward<Args>(args)...)) {
        static auto decorated = make_decorated_function(
            [](auto&&... a) -> decltype(grb::eWiseApply(std::forward<decltype(a)>(a)...)) {
                return grb::eWiseApply(std::forward<decltype(a)>(a)...);
            },
            "eWiseApply"
        );
        return decorated(std::forward<Args>(args)...);
    }

    // Decorated foldl
    template<typename... Args>
    auto decorated_foldl(Args&&... args) 
        -> decltype(grb::foldl(std::forward<Args>(args)...)) {
        static auto decorated = make_decorated_function(
            [](auto&&... a) -> decltype(grb::foldl(std::forward<decltype(a)>(a)...)) {
                return grb::foldl(std::forward<decltype(a)>(a)...);
            },
            "foldl"
        );
        return decorated(std::forward<Args>(args)...);
    }

    // Decorated dot
    template<typename... Args>
    auto decorated_dot(Args&&... args) 
        -> decltype(grb::dot(std::forward<Args>(args)...)) {
        static auto decorated = make_decorated_function(
            [](auto&&... a) -> decltype(grb::dot(std::forward<decltype(a)>(a)...)) {
                return grb::dot(std::forward<decltype(a)>(a)...);
            },
            "dot"
        );
        return decorated(std::forward<Args>(args)...);
    }
}

// Step 3: Replace the original functions with preprocessor macros
// This avoids circular dependencies while keeping the same function names
#define eWiseApply decorated_eWiseApply
#define foldl decorated_foldl
#define dot decorated_dot

#endif // GRB_ENABLE_TRACING

