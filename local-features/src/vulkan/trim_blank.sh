#!/usr/bin/env bash

# Remove blank lines left by gpp #include macros.gpp

awk '
    /\/\/ TRIM_ABOVE/ { 
        if (!found) { 
            print "" 
        } 
        found=1; 
        print 
        next
    } 
    found
    END {
        if (!found) {
            print content
        }
    }
    !found {
        content = content $0 "\n"
    }
' "$@"
