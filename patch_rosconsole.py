#!/usr/bin/env python3
"""Patch rosconsole_log4cxx.cpp for log4cxx 0.12+ (Ubuntu 22.04) compatibility.

log4cxx 0.12 changed all custom ObjectPtr types to std::shared_ptr,
and several functions changed return types from raw pointers to smart pointers.
"""
import re
from pathlib import Path

target = Path(
    '/opt/ros_noetic_ws/src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp'
)
s = target.read_text()
original = s

# 1) g_log4cxx_appender: raw pointer -> shared_ptr (keep subclass type)
s = re.sub(
    r'(static\s+)?Log4cxxAppender\s*\*\s*g_log4cxx_appender\s*=\s*(0|NULL|nullptr)\s*;',
    'static std::shared_ptr<Log4cxxAppender> g_log4cxx_appender;',
    s,
)
s = re.sub(
    r'g_log4cxx_appender\s*=\s*new\s+Log4cxxAppender\((\w+)\)',
    r'g_log4cxx_appender = std::make_shared<Log4cxxAppender>(\1)',
    s,
)
s = re.sub(r'delete\s+g_log4cxx_appender\s*;', 'g_log4cxx_appender.reset();',
           s)

# 2) ROSConsoleStdioAppender: wrap raw pointer in shared_ptr for addAppender()
s = re.sub(
    r'((?:logger|l)->addAppender)\(new\s+ROSConsoleStdioAppender\)',
    r'\1(log4cxx::AppenderPtr(new ROSConsoleStdioAppender))',
    s,
)

# 3) Logger::getLogger() returns shared_ptr<Logger> in 0.12, not Logger*
#    Functions returning void* need .get()
s = re.sub(
    r'return\s+(log4cxx::Logger::getLogger\([^)]+\))\s*;',
    r'return \1.get();',
    s,
)

# 4) getLoggerRepository() returns weak_ptr - need .lock() everywhere
#    Pattern: ...->getLoggerRepository()->...
s = re.sub(
    r'getLoggerRepository\(\)->',
    'getLoggerRepository().lock()->',
    s,
)
#    Pattern: LoggerRepositoryPtr repo = ...getLoggerRepository();
s = re.sub(
    r'(LoggerRepositoryPtr\s+\w+\s*=\s*[^;]*?)getLoggerRepository\(\)',
    r'\1getLoggerRepository().lock()',
    s,
)

# 5) Logger* casts from void* - Logger::getLogger() used to return Logger*
#    static_cast<Logger*>(handle) patterns still work since we store .get()
#    But LoggerPtr(static_cast<Logger*>(...)) patterns may exist
#    No change needed if we patched getHandle to return raw .get()

# 6) getCurrentLoggers() in log4cxx 0.12 returns LoggerList
#    (std::vector<LoggerPtr>), so keep LoggerList usages intact.

if s == original:
    print('WARNING: no changes were made!')
    for i, line in enumerate(original.splitlines()[:100], 1):
        print(f'  {i}: {line}')
else:
    target.write_text(s)
    changes = sum(1 for a, b in zip(original.splitlines(), s.splitlines())
                  if a != b)
    new_lines = len(s.splitlines()) - len(original.splitlines())
    print(
        f'Patched rosconsole_log4cxx.cpp: {changes} lines changed, {new_lines} lines added'
    )
