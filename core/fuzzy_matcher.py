"""
Fuzzy Matcher Utility.

This module provides fuzzy string matching utilities using
Levenshtein distance and Jaro-Winkler similarity algorithms.
Used for matching OCR results against known valid values.
"""

from typing import List, Tuple


class FuzzyMatcher:
    """
    Utility class for fuzzy string matching.
    
    Combines Levenshtein distance and Jaro-Winkler similarity
    algorithms to find the best match for a given string.
    """
    
    @staticmethod
    def levenshteinDistance(a: str, b: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Minimum number of edits needed to transform a into b
        """
        if not a:
            return len(b) if b else 0
        if not b:
            return len(a)
        
        # Use dynamic programming approach
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        
        for i in range(1, len(a) + 1):
            curr[0] = i
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # Deletion
                    curr[j - 1] + 1,  # Insertion
                    prev[j - 1] + cost  # Substitution
                )
            prev, curr = curr, prev
        
        return prev[len(b)]
    
    @staticmethod
    def levenshteinSimilarity(a: str, b: str) -> float:
        """
        Calculate Levenshtein similarity score (0-1).
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Similarity score where 1.0 means identical strings
        """
        if not a and not b:
            return 1.0
        
        dist = FuzzyMatcher.levenshteinDistance(a, b)
        maxLen = max(len(a or ''), len(b or ''))
        
        return 1.0 - dist / maxLen if maxLen > 0 else 1.0
    
    @staticmethod
    def jaroWinklerSimilarity(s: str, t: str, prefixScale: float = 0.1) -> float:
        """
        Calculate Jaro-Winkler similarity score (0-1).
        
        Jaro-Winkler gives higher scores to strings with matching prefixes.
        
        Args:
            s: First string
            t: Second string
            prefixScale: Scaling factor for prefix bonus (default: 0.1)
            
        Returns:
            Similarity score where 1.0 means identical strings
        """
        if not s and not t:
            return 1.0
        if not s or not t:
            return 0.0
        
        if s == t:
            return 1.0
        
        # Calculate matching window
        matchWindow = max(len(s), len(t)) // 2 - 1
        if matchWindow < 0:
            matchWindow = 0
        
        sMatches = [False] * len(s)
        tMatches = [False] * len(t)
        matches = 0
        transpositions = 0
        
        # Find matches within the window
        for i in range(len(s)):
            start = max(0, i - matchWindow)
            end = min(i + matchWindow + 1, len(t))
            
            for j in range(start, end):
                if tMatches[j] or s[i] != t[j]:
                    continue
                sMatches[i] = True
                tMatches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len(s)):
            if not sMatches[i]:
                continue
            while not tMatches[k]:
                k += 1
            if s[i] != t[k]:
                transpositions += 1
            k += 1
        
        # Calculate Jaro similarity
        jaro = (
            matches / len(s) + 
            matches / len(t) + 
            (matches - transpositions / 2) / matches
        ) / 3
        
        # Calculate common prefix (up to 4 characters)
        prefix = 0
        for i in range(min(4, len(s), len(t))):
            if s[i] == t[i]:
                prefix += 1
            else:
                break
        
        # Jaro-Winkler with prefix bonus
        return jaro + prefix * prefixScale * (1 - jaro)
    
    @staticmethod
    def combinedSimilarity(a: str, b: str) -> float:
        """
        Calculate combined similarity using max of Levenshtein and Jaro-Winkler.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            Maximum similarity score from both algorithms
        """
        # Normalize strings for comparison
        na = (a or '').strip().upper()
        nb = (b or '').strip().upper()
        
        lev = FuzzyMatcher.levenshteinSimilarity(na, nb)
        jw = FuzzyMatcher.jaroWinklerSimilarity(na, nb)
        
        return max(lev, jw)
    
    @staticmethod
    def bestMatch(
        text: str, 
        candidates: List[str], 
        minScore: float = 0.0
    ) -> Tuple[str, float]:
        """
        Find the best matching candidate for a given text.
        
        Args:
            text: Text to match
            candidates: List of valid candidate strings
            minScore: Minimum score to consider a match (default: 0.0)
            
        Returns:
            Tuple of (best_match, score). Returns ("", 0.0) if no match found.
        """
        if not text or not candidates:
            return ("", 0.0)
        
        best = ""
        bestScore = minScore
        
        for candidate in candidates:
            score = FuzzyMatcher.combinedSimilarity(text, candidate)
            if score > bestScore:
                bestScore = score
                best = candidate
        
        return (best, bestScore)
    
    @staticmethod
    def isMatch(text: str, target: str, threshold: float = 0.8) -> bool:
        """
        Check if text matches target with given threshold.
        
        Args:
            text: Text to check
            target: Target to match against
            threshold: Minimum similarity score (default: 0.8)
            
        Returns:
            True if similarity score >= threshold
        """
        return FuzzyMatcher.combinedSimilarity(text, target) >= threshold
