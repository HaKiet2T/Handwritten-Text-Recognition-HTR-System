"""
Enhanced Post-Processing for Web App - Integrates Phase 3 Improvements
Replaces basic spell checker with OCR-aware correction
"""
import re


class EnhancedOCRCorrector:
    """
    OCR-aware spell checker with common OCR confusions
    Based on Phase 3 implementation but simplified for web integration
    """
    
    def __init__(self):
        # Common OCR confusions (digit/letter pairs)
        self.ocr_confusions = {
            '0': 'o',  # Zero → letter O
            'O': 'o',  # Capital O → lowercase o
            '1': 'l',  # One → letter l
            'I': 'l',  # Capital I → lowercase l
            '5': 's',  # Five → letter s
            'S': 's',  # Capital S → lowercase s
            '8': 'B',  # Eight → letter B
            '6': 'b',  # Six → letter b
            '2': 'z',  # Two → letter z
            '9': 'g',  # Nine → letter g
        }
        
        # Common English words (dictionary)
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self):
        """Load common English words"""
        # Extended dictionary - 200+ common words
        common_words = {
            # Top 100 English words
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its',
            'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our',
            'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
            
            # Common technical/CS words
            'hello', 'world', 'computer', 'vision', 'machine', 'learning',
            'deep', 'neural', 'network', 'model', 'data', 'test', 'image',
            'recognition', 'handwriting', 'artificial', 'intelligence',
            'application', 'system', 'algorithm', 'python', 'code', 'program',
            'software', 'hardware', 'internet', 'web', 'app', 'api', 'database',
            'server', 'client', 'cloud', 'digital', 'technology', 'science',
            'function', 'class', 'method', 'variable', 'array', 'string', 'number',
            'object', 'process', 'memory', 'storage', 'network', 'security',
            
            # Common verbs
            'write', 'read', 'open', 'close', 'start', 'stop', 'run', 'test',
            'create', 'update', 'delete', 'save', 'load', 'print', 'show',
            'send', 'receive', 'connect', 'download', 'upload', 'install',
            'build', 'compile', 'execute', 'debug', 'deploy', 'configure',
            
            # Common nouns
            'file', 'folder', 'document', 'text', 'word', 'line', 'page',
            'screen', 'window', 'button', 'menu', 'icon', 'link', 'email',
            'message', 'user', 'admin', 'password', 'login', 'account',
            'name', 'address', 'phone', 'date', 'time', 'place', 'thing',
            
            # Numbers as words
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
            'eight', 'nine', 'ten', 'eleven', 'twelve', 'twenty', 'thirty',
            'hundred', 'thousand', 'million', 'billion',
            
            # Common adjectives
            'good', 'bad', 'big', 'small', 'new', 'old', 'high', 'low',
            'fast', 'slow', 'hot', 'cold', 'long', 'short', 'easy', 'hard',
            'right', 'wrong', 'true', 'false', 'full', 'empty', 'open', 'close',
        }
        return common_words
    
    def correct_ocr_confusions(self, text):
        """
        Apply OCR confusion fixes
        Example: "hell0" → "hello", "w0rld" → "world"
        """
        corrected = text
        for wrong, right in self.ocr_confusions.items():
            corrected = corrected.replace(wrong, right)
        return corrected
    
    def correct_word(self, word):
        """Correct a single word"""
        # Convert to lowercase for dictionary lookup
        word_lower = word.lower()
        
        # If already in dictionary, return as-is
        if word_lower in self.dictionary:
            return word
        
        # Try OCR confusion fixes
        fixed = self.correct_ocr_confusions(word)
        if fixed.lower() in self.dictionary:
            return fixed
        
        # Find close matches (edit distance 1-2)
        best_match = self._find_closest_match(fixed.lower())
        if best_match:
            # Preserve original case pattern
            return self._apply_case_pattern(word, best_match)
        
        # No correction found, return fixed version
        return fixed
    
    def correct(self, text):
        """
        Correct entire text
        
        Args:
            text: Raw OCR output
        
        Returns:
            Corrected text
        """
        if not text or not text.strip():
            return text
        
        # Split into words while preserving spaces
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Keep punctuation
            prefix = ''
            suffix = ''
            
            # Extract leading punctuation
            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]
            
            # Extract trailing punctuation
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]
            
            # Correct the word
            if word:
                corrected = self.correct_word(word)
                corrected_words.append(prefix + corrected + suffix)
            else:
                corrected_words.append(prefix + suffix)
        
        return ' '.join(corrected_words)
    
    def _find_closest_match(self, word):
        """Find closest dictionary match (edit distance 1-2)"""
        # Check edit distance 1
        for dict_word in self.dictionary:
            if self._edit_distance(word, dict_word) == 1:
                return dict_word
        
        # Check edit distance 2 (more expensive)
        for dict_word in self.dictionary:
            if self._edit_distance(word, dict_word) == 2:
                return dict_word
        
        return None
    
    def _edit_distance(self, s1, s2):
        """Calculate edit distance (Levenshtein)"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances
        
        return distances[-1]
    
    def _apply_case_pattern(self, original, corrected):
        """Apply original case pattern to corrected word"""
        if not original:
            return corrected
        
        # All uppercase
        if original.isupper():
            return corrected.upper()
        
        # First letter uppercase
        if original[0].isupper():
            return corrected.capitalize()
        
        # All lowercase
        return corrected.lower()


# Global corrector instance
_corrector = None


def get_enhanced_corrector():
    """Get or create enhanced corrector instance"""
    global _corrector
    if _corrector is None:
        _corrector = EnhancedOCRCorrector()
    return _corrector


def correct_prediction(text):
    """
    Correct OCR prediction with Phase 3 enhancements
    
    Args:
        text: Raw model prediction
    
    Returns:
        Corrected text
    """
    corrector = get_enhanced_corrector()
    return corrector.correct(text)


if __name__ == "__main__":
    # Test the corrector
    print("="*70)
    print("🧪 TESTING ENHANCED OCR CORRECTOR (Phase 3)")
    print("="*70)
    
    corrector = EnhancedOCRCorrector()
    
    test_cases = [
        ("hell0 w0rld", "hello world"),
        ("c0mputer v1sion", "computer vision"),
        ("handwr1t1ng rec0gnit1on", "handwriting recognition"),
        ("artif1cial intel1igence", "artificial intelligence"),
        ("mach1ne learn1ng", "machine learning"),
        ("deep neur0nal netw0rk", "deep neuronal network"),
        ("8usiness appl1cat1on", "Business application"),
        ("test1ng the c0de", "testing the code"),
    ]
    
    print("\n📝 Test Cases:\n")
    
    correct_count = 0
    for raw, expected in test_cases:
        corrected = corrector.correct(raw)
        is_correct = corrected.lower() == expected.lower()
        status = "✅" if is_correct else "❌"
        
        if is_correct:
            correct_count += 1
        
        print(f"{status} {raw:35} → {corrected:35} (expected: {expected})")
    
    accuracy = (correct_count / len(test_cases)) * 100
    
    print("\n" + "="*70)
    print(f"✅ Accuracy: {correct_count}/{len(test_cases)} ({accuracy:.0f}%)")
    print("="*70)
    print("\n💡 To integrate with web app:")
    print("   from src.postprocessing.enhanced_corrector import correct_prediction")
    print("   corrected_text = correct_prediction(raw_prediction)")
