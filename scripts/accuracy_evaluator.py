"""
Accuracy Evaluator Module.

This module provides functionality to evaluate OCR accuracy
by comparing OCR results against ground truth data using
fuzzy string matching algorithms.

Usage:
    from scripts.accuracy_evaluator import AccuracyEvaluator
    
    evaluator = AccuracyEvaluator()
    results = evaluator.evaluate_pipeline(samples_dir, results_dir)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.fuzzy_matcher import FuzzyMatcher

# Configure logging
logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """
    Evaluate OCR accuracy against ground truth using fuzzy matching.
    
    Uses a two-phase matching algorithm:
    1. Phase 1: Confident matching (similarity >= 0.90)
    2. Phase 2: Optimal matching for remaining fields
    
    Attributes:
        FIELD_PRIORITY: Order of fields to process in phase 1
        FIELD_WEIGHTS: Weight of each field for overall score calculation
        THRESHOLD_EXACT: Threshold for exact match (1.0)
        THRESHOLD_ACCEPTABLE: Threshold for acceptable match (0.90)
    """
    
    # Field processing order (phase 1)
    FIELD_PRIORITY = ['color', 'productCode', 'size', 'positionQuantity']
    
    # Weight for each field (total = 1.0)
    FIELD_WEIGHTS = {
        'color': 0.25,
        'productCode': 0.25,
        'size': 0.25,
        'positionQuantity': 0.25
    }
    
    # Match classification thresholds
    THRESHOLD_EXACT = 1.0
    THRESHOLD_ACCEPTABLE = 0.90
    
    def __init__(self):
        """Initialize the accuracy evaluator."""
        self.fuzzyMatcher = FuzzyMatcher()
    
    @staticmethod
    def normalizeText(text: str) -> str:
        """
        Normalize text for comparison.
        
        Converts to uppercase and removes all whitespace.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text (uppercase, no spaces)
        """
        if not text:
            return ""
        return text.strip().upper().replace(" ", "")
    
    def calculateSimilarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts after normalization.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        normalized1 = self.normalizeText(text1)
        normalized2 = self.normalizeText(text2)
        
        if not normalized1 and not normalized2:
            return 1.0
        if not normalized1 or not normalized2:
            return 0.0
        
        # Use combined similarity (max of Levenshtein and Jaro-Winkler)
        return FuzzyMatcher.combinedSimilarity(normalized1, normalized2)
    
    def classifyMatch(self, similarity: float, hasText: bool) -> str:
        """
        Classify match type based on similarity score.
        
        Args:
            similarity: Similarity score (0.0 - 1.0)
            hasText: Whether OCR text was found
            
        Returns:
            Match type: 'exact', 'acceptable', 'rejected', or 'missing'
        """
        if not hasText:
            return "missing"
        if similarity >= self.THRESHOLD_EXACT:
            return "exact"
        elif similarity >= self.THRESHOLD_ACCEPTABLE:
            return "acceptable"
        else:
            return "rejected"
    
    def findBestMatch(
        self,
        target: str,
        availableTexts: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching OCR text for a target string.
        
        Args:
            target: Ground truth text to match
            availableTexts: List of OCR text dicts with 'text' and 'score' keys
            
        Returns:
            Tuple of (best matching text dict or None, similarity score)
        """
        if not target or not availableTexts:
            return (None, 0.0)
        
        bestMatch = None
        bestSimilarity = 0.0
        
        for textDict in availableTexts:
            ocrText = textDict.get('text', '')
            similarity = self.calculateSimilarity(target, ocrText)
            
            if similarity > bestSimilarity:
                bestSimilarity = similarity
                bestMatch = textDict
        
        return (bestMatch, bestSimilarity)
    
    def phase1ConfidentMatching(
        self,
        groundTruth: Dict,
        availableTexts: List[Dict]
    ) -> Tuple[Dict, List[str], List[Dict]]:
        """
        Phase 1: Match fields with similarity >= 0.90.
        
        Processes fields in priority order and matches only confident ones.
        
        Args:
            groundTruth: Ground truth data with field values
            availableTexts: List of available OCR text dicts
            
        Returns:
            Tuple of:
                - matched_results: Dict of matched fields
                - remaining_fields: List of unmatched field names
                - remaining_texts: List of unused OCR text dicts
        """
        matchedResults = {}
        remainingFields = list(self.FIELD_PRIORITY)
        remainingTexts = list(availableTexts)
        
        for field in self.FIELD_PRIORITY:
            gtValue = groundTruth.get(field, '')
            
            if not gtValue:
                # No ground truth for this field
                matchedResults[field] = {
                    'gt_value': '',
                    'ocr_value': '',
                    'similarity': 0.0,
                    'match_type': 'missing'
                }
                remainingFields.remove(field)
                continue
            
            bestMatch, similarity = self.findBestMatch(gtValue, remainingTexts)
            
            if similarity >= self.THRESHOLD_ACCEPTABLE:
                # Confident match found
                ocrValue = bestMatch.get('text', '') if bestMatch else ''
                matchType = self.classifyMatch(similarity, True)
                
                matchedResults[field] = {
                    'gt_value': gtValue,
                    'ocr_value': ocrValue,
                    'similarity': similarity,
                    'match_type': matchType
                }
                
                remainingFields.remove(field)
                if bestMatch in remainingTexts:
                    remainingTexts.remove(bestMatch)
                
                logger.debug(
                    f"Phase 1: {field} matched '{ocrValue}' "
                    f"(sim={similarity:.3f}, type={matchType})"
                )
        
        return (matchedResults, remainingFields, remainingTexts)
    
    def phase2OptimalMatching(
        self,
        groundTruth: Dict,
        remainingFields: List[str],
        remainingTexts: List[Dict]
    ) -> Dict:
        """
        Phase 2: Optimal matching for remaining fields.
        
        Finds the best (field, text) pair with highest similarity
        at each iteration until all fields are matched or no texts remain.
        
        Args:
            groundTruth: Ground truth data
            remainingFields: List of unmatched field names
            remainingTexts: List of unused OCR text dicts
            
        Returns:
            Dict of matched results for remaining fields
        """
        matchedResults = {}
        fieldsToMatch = list(remainingFields)
        textsToUse = list(remainingTexts)
        
        while fieldsToMatch and textsToUse:
            # Calculate all (field, text) pairs with similarity
            allPairs = []
            
            for field in fieldsToMatch:
                gtValue = groundTruth.get(field, '')
                if not gtValue:
                    continue
                    
                for textDict in textsToUse:
                    ocrText = textDict.get('text', '')
                    similarity = self.calculateSimilarity(gtValue, ocrText)
                    allPairs.append({
                        'field': field,
                        'text_dict': textDict,
                        'similarity': similarity
                    })
            
            if not allPairs:
                break
            
            # Find the best pair
            bestPair = max(allPairs, key=lambda x: x['similarity'])
            bestField = bestPair['field']
            bestTextDict = bestPair['text_dict']
            bestSimilarity = bestPair['similarity']
            
            ocrValue = bestTextDict.get('text', '')
            matchType = self.classifyMatch(bestSimilarity, True)
            
            matchedResults[bestField] = {
                'gt_value': groundTruth.get(bestField, ''),
                'ocr_value': ocrValue,
                'similarity': bestSimilarity,
                'match_type': matchType
            }
            
            fieldsToMatch.remove(bestField)
            textsToUse.remove(bestTextDict)
            
            logger.debug(
                f"Phase 2: {bestField} matched '{ocrValue}' "
                f"(sim={bestSimilarity:.3f}, type={matchType})"
            )
        
        # Handle fields with no text to assign
        for field in fieldsToMatch:
            gtValue = groundTruth.get(field, '')
            matchedResults[field] = {
                'gt_value': gtValue,
                'ocr_value': '',
                'similarity': 0.0,
                'match_type': 'missing'
            }
            logger.debug(f"Phase 2: {field} has no text to match (missing)")
        
        return matchedResults
    
    def evaluateImage(
        self,
        groundTruth: Dict,
        ocrResult: Dict
    ) -> Dict:
        """
        Evaluate OCR accuracy for a single image.
        
        Uses two-phase matching algorithm:
        1. Phase 1: Confident matching (similarity >= 0.90)
        2. Phase 2: Optimal matching for remaining fields
        
        Args:
            groundTruth: Ground truth data with field values
            ocrResult: OCR result with text_regions list
            
        Returns:
            Dict containing:
                - Field-level results (gt_value, ocr_value, similarity, match_type)
                - avg_score: Weighted average score
                - all_exact: Whether all fields are exact matches
                - all_acceptable: Whether all fields are acceptable
        """
        # Extract OCR text regions
        textRegions = ocrResult.get('text_regions', [])
        availableTexts = [
            {'text': region.get('text', ''), 'score': region.get('score', 0.0)}
            for region in textRegions
        ]
        
        # Phase 1: Confident matching
        phase1Results, remainingFields, remainingTexts = self.phase1ConfidentMatching(
            groundTruth, availableTexts
        )
        
        # Phase 2: Optimal matching for remaining
        phase2Results = self.phase2OptimalMatching(
            groundTruth, remainingFields, remainingTexts
        )
        
        # Merge results
        allResults = {**phase1Results, **phase2Results}
        
        # Calculate weighted average score
        avgScore = 0.0
        for field, weight in self.FIELD_WEIGHTS.items():
            fieldResult = allResults.get(field, {})
            similarity = fieldResult.get('similarity', 0.0)
            avgScore += similarity * weight
        
        # Check if all fields are exact/acceptable
        allExact = all(
            allResults.get(f, {}).get('match_type') == 'exact'
            for f in self.FIELD_PRIORITY
        )
        allAcceptable = all(
            allResults.get(f, {}).get('match_type') in ['exact', 'acceptable']
            for f in self.FIELD_PRIORITY
        )
        
        return {
            'fields': allResults,
            'avg_score': avgScore,
            'all_exact': allExact,
            'all_acceptable': allAcceptable
        }
    
    def loadGroundTruth(self, filePath: Path) -> Optional[Dict]:
        """
        Load ground truth data from JSON file.
        
        Args:
            filePath: Path to ground truth JSON file
            
        Returns:
            Ground truth dict or None if file not found
        """
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Ground truth not found: {filePath}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filePath}: {e}")
            return None
    
    def loadOcrResult(self, filePath: Path) -> Optional[Dict]:
        """
        Load OCR result from JSON file.
        
        Args:
            filePath: Path to OCR result JSON file
            
        Returns:
            OCR result dict or None if file not found
        """
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"OCR result not found: {filePath}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filePath}: {e}")
            return None
    
    def extractFilenameBase(self, filename: str) -> str:
        """
        Extract base filename for matching between ground truth and OCR results.
        
        Ground truth: result_components_frame_xxx.json
        OCR result: components_frame_xxx_result.json
        
        Args:
            filename: Input filename
            
        Returns:
            Base filename (e.g., 'components_frame_xxx')
        """
        name = Path(filename).stem
        
        # Remove 'result_' prefix (ground truth format)
        if name.startswith('result_'):
            name = name[7:]
        
        # Remove '_result' suffix (OCR result format)
        if name.endswith('_result'):
            name = name[:-7]
        
        return name
    
    def evaluatePipeline(
        self,
        samplesDir: Path,
        resultsDir: Path,
        pipelineName: str = "Pipeline"
    ) -> pd.DataFrame:
        """
        Evaluate accuracy for an entire pipeline.
        
        Args:
            samplesDir: Directory containing ground truth JSON files
            resultsDir: Directory containing OCR result JSON files
            pipelineName: Name of the pipeline (for logging)
            
        Returns:
            DataFrame with accuracy results for each image
        """
        results = []
        
        # Get all ground truth files
        gtFiles = list(samplesDir.glob("result_*.json"))
        logger.info(f"Found {len(gtFiles)} ground truth files in {samplesDir}")
        
        for gtFile in gtFiles:
            baseFilename = self.extractFilenameBase(gtFile.name)
            
            # Find corresponding OCR result
            ocrFilename = f"{baseFilename}_result.json"
            ocrFile = resultsDir / ocrFilename
            
            if not ocrFile.exists():
                logger.warning(f"OCR result not found for {baseFilename}")
                continue
            
            # Load data
            groundTruth = self.loadGroundTruth(gtFile)
            ocrResult = self.loadOcrResult(ocrFile)
            
            if groundTruth is None or ocrResult is None:
                continue
            
            # Evaluate
            evalResult = self.evaluateImage(groundTruth, ocrResult)
            
            # Build result row
            row = {
                'filename': f"{baseFilename}.png",
                'pipeline': pipelineName
            }
            
            # Add field-level results
            for field in self.FIELD_PRIORITY:
                fieldResult = evalResult['fields'].get(field, {})
                row[f'gt_{field}'] = fieldResult.get('gt_value', '')
                row[f'ocr_{field}'] = fieldResult.get('ocr_value', '')
                row[f'score_{field}'] = fieldResult.get('similarity', 0.0)
                row[f'match_type_{field}'] = fieldResult.get('match_type', 'missing')
            
            # Add overall metrics
            row['avg_accuracy_score'] = evalResult['avg_score']
            row['all_exact'] = evalResult['all_exact']
            row['all_acceptable'] = evalResult['all_acceptable']
            
            results.append(row)
        
        logger.info(f"Evaluated {len(results)} images for {pipelineName}")
        
        return pd.DataFrame(results)
    
    def calculateSummaryStatistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from evaluation results.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            Dict with summary statistics
        """
        if df.empty:
            return {}
        
        stats = {
            'num_images': len(df),
            'fields': {}
        }
        
        for field in self.FIELD_PRIORITY:
            scoreCol = f'score_{field}'
            matchTypeCol = f'match_type_{field}'
            
            if scoreCol not in df.columns:
                continue
            
            fieldStats = {
                'avg_score': df[scoreCol].mean(),
                'exact_count': (df[matchTypeCol] == 'exact').sum(),
                'acceptable_count': (df[matchTypeCol].isin(['exact', 'acceptable'])).sum(),
                'rejected_count': (df[matchTypeCol] == 'rejected').sum(),
                'missing_count': (df[matchTypeCol] == 'missing').sum(),
                'exact_pct': (df[matchTypeCol] == 'exact').mean() * 100,
                'acceptable_pct': (df[matchTypeCol].isin(['exact', 'acceptable'])).mean() * 100
            }
            
            stats['fields'][field] = fieldStats
        
        # Overall statistics
        stats['overall'] = {
            'avg_score': df['avg_accuracy_score'].mean(),
            'all_exact_count': df['all_exact'].sum(),
            'all_acceptable_count': df['all_acceptable'].sum(),
            'all_exact_pct': df['all_exact'].mean() * 100,
            'all_acceptable_pct': df['all_acceptable'].mean() * 100
        }
        
        return stats
