# NutriSchema Product Requirements Document (PRD)

## 1. Executive Summary & Vision

* **Product Name:** **NutriSchema** (`nutrischema`)
* **Tagline:** *The Universal Data Model & Parser Suite for Computational Nutrition and AI Research*
* **Target Audience:**
  * Preventive medicine and Whole Health researchers (e.g., Alice L. Walton School of Medicine - AWSOM).
  * Computational nutritionists, digital health researchers, and AI developers building metabolic, microbiome, or dietary models.
* **Core Purpose:**
  * Eliminate the data-wrangling bottleneck in nutrition AI research by providing a normalized, unit-standardized JSON Schema and Pydantic data model.
  * Automate cross-database connector generation (USDA, MEXT Japan, EFSA) using **Factory.ai (Droid)**.
  * Deliver ML-ready outputs (PyTorch tensors, Pandas DataFrames, Parquet) for immediate model training.

---

## 2. Target Personas & Use Cases


```

[AI/Bio-Researcher]                   [Preventive Medicine Scholar]              [Factory.ai / Droid Pipeline]
├── ML-Ready Tensor Export            ├── Whole Health & Metabolic Indexes       ├── Automated Parser Generation
├── Unified SI Unit Conversion        ├── NOVA Classification (NOVA 1-4)         ├── CI/CD Test Suite Generation
└── Multi-DB Normalization            └── Glycemic Index/Load Schema            └── Schema Versioning & Patching

```

### ① Computational Nutrition & AI Researcher
* **Needs:** Access clean, standardized nutritional vectors without manually parsing disparate regional databases (USDA FDC, MEXT Japan, EuroFIR). Requires direct export to PyTorch/Pandas.

### ② Preventive Medicine & Metabolic Scholar (AWSOM / Heartland Institutes)
* **Needs:** Model holistic health parameters beyond simple macronutrients—including glycemic index, glycemic load, NOVA ultra-processed food classification, and polyphenol contents.

### ③ Automated OSS Developer (Factory.ai Droid Workflow)
* **Needs:** Clear, strongly-typed Python models (`Pydantic v2`) and schema specifications (`JSON Schema Draft 2020-12`) that allow autonomous AI coding agents to generate, test, and maintain parser connectors.

---

## 3. Core System Architecture


```

[Global Nutrition Databases]
├── USDA FoodData Central (USA) ──┐
├── MEXT Food Composition (JP)  ├──> [1. NutriSchema Parser Pipeline (Factory Droid Generated)]
└── EFSA EuroFIR (EU)          ──┘     - Normalizes unit variations (mg, mcg, IU) to SI Standards
- Maps regional IDs to `NutriSchema Universal Food ID`
│
▼
[2. NutriSchema Core Data Engine]
- Validated via Pydantic v2 & JSON Schema Draft 2020-12
- Metabolic & Whole Health Index Enrichment
│
▼
[3. Research Export Layer]
- Pandas DataFrame / Parquet Exporter
- PyTorch / TensorFlow Tensor Exporter
- Open Research API & CLI (`pip install nutrischema`)

```

---

## 4. Universal Data Specification (JSON Schema)

```json
{
  "$schema": "[https://json-schema.org/draft/2020-12/schema](https://json-schema.org/draft/2020-12/schema)",
  "$id": "[https://schema.nutrischema.org/v1/food-item.json](https://schema.nutrischema.org/v1/food-item.json)",
  "title": "NutriSchemaFoodItem",
  "description": "Universal data model for nutritional, metabolic, and preventive health AI research.",
  "type": "object",
  "required": [
    "foodId",
    "canonicalName",
    "sourceDatabase",
    "servingSizeGrams",
    "macronutrients",
    "micronutrients"
  ],
  "properties": {
    "foodId": { "type": "string", "example": "USDA_FDC_170379" },
    "canonicalName": { "type": "string", "example": "Broccoli, raw" },
    "scientificName": { "type": "string", "example": "Brassica oleracea var. italica" },
    "sourceDatabase": {
      "type": "string",
      "enum": ["USDA_FDC", "MEXT_JAPAN", "EFSA_EUROFIR", "CUSTOM_ACADEMIC"]
    },
    "servingSizeGrams": { "type": "number", "default": 100.0 },
    "macronutrients": {
      "type": "object",
      "required": ["energyKcal", "proteinGrams", "fatGrams", "carbohydratesGrams"],
      "properties": {
        "energyKcal": { "type": "number", "minimum": 0 },
        "proteinGrams": { "type": "number", "minimum": 0 },
        "fatGrams": { "type": "number", "minimum": 0 },
        "carbohydratesGrams": { "type": "number", "minimum": 0 },
        "fiberGrams": { "type": "number", "minimum": 0 },
        "sugarGrams": { "type": "number", "minimum": 0 }
      }
    },
    "micronutrients": {
      "type": "object",
      "properties": {
        "vitamins": {
          "type": "object",
          "properties": {
            "vitaminAMcgRAE": { "type": "number", "minimum": 0 },
            "vitaminCMg": { "type": "number", "minimum": 0 },
            "vitaminDMcg": { "type": "number", "minimum": 0 },
            "vitaminEMg": { "type": "number", "minimum": 0 }
          }
        },
        "minerals": {
          "type": "object",
          "properties": {
            "calciumMg": { "type": "number", "minimum": 0 },
            "ironMg": { "type": "number", "minimum": 0 },
            "potassiumMg": { "type": "number", "minimum": 0 },
            "sodiumMg": { "type": "number", "minimum": 0 }
          }
        }
      }
    },
    "metabolicIndexes": {
      "type": "object",
      "properties": {
        "glycemicIndex": { "type": "number", "minimum": 0, "maximum": 100 },
        "glycemicLoad": { "type": "number", "minimum": 0 },
        "ultraProcessedFoodScore": { "type": "integer", "minimum": 1, "maximum": 4 },
        "polyphenolsMg": { "type": "number", "minimum": 0 }
      }
    }
  }
}

```

---

## 5. Technical Implementation (Pydantic v2 Core Model)

```python
"""
NutriSchema Core Models (nutrischema/models.py)
"""

from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SourceDatabase(str, Enum):
    USDA_FDC = "USDA_FDC"
    MEXT_JAPAN = "MEXT_JAPAN"
    EFSA_EUROFIR = "EFSA_EUROFIR"
    CUSTOM_ACADEMIC = "CUSTOM_ACADEMIC"


class Macronutrients(BaseModel):
    energyKcal: float = Field(ge=0, description="Energy in kilocalories")
    proteinGrams: float = Field(ge=0, description="Protein in grams")
    fatGrams: float = Field(ge=0, description="Total lipid (fat) in grams")
    carbohydratesGrams: float = Field(ge=0, description="Carbohydrates in grams")
    fiberGrams: Optional[float] = Field(default=0.0, ge=0, description="Dietary fiber in grams")
    sugarGrams: Optional[float] = Field(default=0.0, ge=0, description="Total sugars in grams")


class Vitamins(BaseModel):
    vitaminAMcgRAE: Optional[float] = Field(default=0.0, ge=0)
    vitaminCMg: Optional[float] = Field(default=0.0, ge=0)
    vitaminDMcg: Optional[float] = Field(default=0.0, ge=0)
    vitaminEMg: Optional[float] = Field(default=0.0, ge=0)


class Minerals(BaseModel):
    calciumMg: Optional[float] = Field(default=0.0, ge=0)
    ironMg: Optional[float] = Field(default=0.0, ge=0)
    potassiumMg: Optional[float] = Field(default=0.0, ge=0)
    sodiumMg: Optional[float] = Field(default=0.0, ge=0)


class Micronutrients(BaseModel):
    vitamins: Optional[Vitamins] = Field(default_factory=Vitamins)
    minerals: Optional[Minerals] = Field(default_factory=Minerals)


class MetabolicIndexes(BaseModel):
    glycemicIndex: Optional[float] = Field(default=None, ge=0, le=100)
    glycemicLoad: Optional[float] = Field(default=None, ge=0)
    ultraProcessedFoodScore: Optional[int] = Field(default=None, ge=1, le=4, description="NOVA Classification 1-4")
    polyphenolsMg: Optional[float] = Field(default=None, ge=0)


class NutriFoodItem(BaseModel):
    foodId: str
    canonicalName: str
    scientificName: Optional[str] = None
    sourceDatabase: SourceDatabase
    servingSizeGrams: float = 100.0
    macronutrients: Macronutrients
    micronutrients: Micronutrients = Field(default_factory=Micronutrients)
    metabolicIndexes: Optional[MetabolicIndexes] = Field(default_factory=MetabolicIndexes)

    def to_pandas_row(self) -> Dict[str, Any]:
        """Convert FoodItem into a flat dictionary ready for Pandas DataFrames or PyTorch tensors."""
        return {
            "food_id": self.foodId,
            "name": self.canonicalName,
            "source": self.sourceDatabase.value,
            "energy_kcal": self.macronutrients.energyKcal,
            "protein_g": self.macronutrients.proteinGrams,
            "fat_g": self.macronutrients.fatGrams,
            "carbs_g": self.macronutrients.carbohydratesGrams,
            "fiber_g": self.macronutrients.fiberGrams,
            "vitamin_c_mg": self.micronutrients.vitamins.vitaminCMg if self.micronutrients.vitamins else 0.0,
            "glycemic_index": self.metabolicIndexes.glycemicIndex if self.metabolicIndexes else None,
            "nova_score": self.metabolicIndexes.ultraProcessedFoodScore if self.metabolicIndexes else None,
        }

```

---

## 6. Factory.ai (Droid) Development Roadmap

1. **Step 1: Base Schema & Models Enforcement**
* Commit `nutrischema/models.py` and `schemas/v1/food-item.json` to GitHub repository.


2. **Step 2: Automated Parser Missions (Factory Droid)**
* Deploy Droids to construct `parsers/usda_fdc.py`, `parsers/mext_japan.py`, and `parsers/efsa_eurofir.py` concurrently.


3. **Step 3: Test Suite & CI/CD Pipeline Generation**
* Utilize Droids to generate unit test coverage exceeding 95% with synthetic regional payload mocks.



```

<FollowUp label="Want to generate the AGENTS.md guide for Factory.ai Droid or drafting the Medium technical post?" query="Please generate the AGENTS.md guidelines for Factory.ai Droid and a draft of the Medium article for NutriSchema."/>

```
