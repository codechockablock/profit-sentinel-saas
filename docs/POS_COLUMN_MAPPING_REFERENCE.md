# Multi-Vendor POS Column Mapping Reference

> **Comprehensive column name → normalized schema mapping for 20+ POS systems.**
> Used by auto-detecting adapters to ingest inventory, sales, and PO data from any
> hardware retail POS system without manual configuration.
>
> **Last updated:** 2026-02-06
> **Source of truth:** `config/pos_mappings/standard_fields.yaml` (400+ aliases)

---

## Table of Contents

1. [Normalized Schema](#1-normalized-schema)
2. [Tier 1: Hardware-Specific POS Systems](#2-tier-1-hardware-specific-pos-systems)
   - [Paladin POS](#21-paladin-pos)
   - [Epicor Eagle](#22-epicor-eagle)
   - [Epicor BisTrack](#23-epicor-bistrack)
   - [Spruce POS](#24-spruce-pos)
   - [NCR Counterpoint](#25-ncr-counterpoint)
   - [POSIM](#26-posim)
   - [IdoSoft POS](#27-idosoft-pos)
3. [Tier 2: General Retail POS Systems](#3-tier-2-general-retail-pos-systems)
   - [Lightspeed R-Series](#31-lightspeed-r-series)
   - [Lightspeed X-Series (Vend)](#32-lightspeed-x-series-vend)
   - [Retail Pro](#33-retail-pro)
   - [KORONA POS](#34-korona-pos)
   - [Rain POS](#35-rain-pos)
   - [MicroBiz](#36-microbiz)
   - [Celerant Command Retail](#37-celerant-command-retail)
   - [Shopify POS](#38-shopify-pos)
   - [Square POS](#39-square-pos)
   - [Clover POS](#310-clover-pos)
   - [Microsoft Dynamics RMS](#311-microsoft-dynamics-rms)
4. [Tier 3: Co-op / Distributor Systems](#4-tier-3-co-op--distributor-systems)
   - [Do It Best](#41-do-it-best)
   - [Ace Hardware (ARS)](#42-ace-hardware-ars)
   - [True Value](#43-true-value)
   - [Orgill](#44-orgill)
5. [EDI Standards (ANSI X12)](#5-edi-standards-ansi-x12)
6. [Auto-Detection Fingerprints](#6-auto-detection-fingerprints)
7. [Format Specifications](#7-format-specifications)
8. [Known Quirks & Caveats](#8-known-quirks--caveats)

---

## 1. Normalized Schema

All POS data maps to these canonical fields. Field importance ratings (1-5) drive
prioritization in column mapping and profit leak analysis.

| Canonical Field       | Rating | Type    | Description                              |
|-----------------------|--------|---------|------------------------------------------|
| `sku`                 | 3      | string  | Primary item identifier                  |
| `description`         | 2      | string  | Item name / description                  |
| `quantity`            | 5      | integer | Quantity on hand (may be negative)       |
| `cost`                | 5      | float   | Unit cost (avg, last, or market)         |
| `revenue`             | 5      | float   | Retail / selling price                   |
| `sold`                | 5      | integer | Units sold (YTD, MTD, or period)         |
| `margin`              | 5      | float   | Profit margin percentage                 |
| `qty_difference`      | 5      | integer | Inventory variance / shrinkage           |
| `inventoried_qty`     | 4      | integer | Physical count quantity                  |
| `sub_total`           | 4      | float   | Extended value (qty × cost or price)     |
| `last_sale_date`      | 4      | date    | Date of most recent sale                 |
| `last_purchase_date`  | 4      | date    | Date of most recent receipt              |
| `vendor`              | 3      | string  | Vendor / supplier name or code           |
| `category`            | 3      | string  | Department / category / class            |
| `status`              | 3      | string  | Active, Inactive, Discontinued           |
| `reorder_point`       | 3      | integer | Min stock / reorder trigger              |
| `package_qty`         | 3      | integer | Pack size / case qty / UOM               |
| `location`            | 3      | string  | Store / warehouse / bin                  |
| `date`                | 2      | date    | Transaction date                         |
| `transaction_id`      | 2      | string  | Receipt / invoice number                 |
| `customer_id`         | 2      | string  | Customer / loyalty ID                    |
| `discount`            | 2      | float   | Discount amount or percentage            |
| `tax`                 | 2      | float   | Tax amount or rate                       |
| `return_flag`         | 2      | boolean | Return / void / credit flag              |

---

## 2. Tier 1: Hardware-Specific POS Systems

### 2.1 Paladin POS

**Vendor:** Paladin Data Corporation
**Market:** Independent hardware, lumber, paint stores
**Confidence:** VERY HIGH (sample store adapter fully implemented)

#### Format Details

| Property         | Value                                                |
|------------------|------------------------------------------------------|
| File format      | CSV (comma) or TSV (tab-delimited)                   |
| Encoding         | UTF-8 or Windows-1252                                |
| Header rows      | 1                                                    |
| Date format      | `YYYYMMDD` (native) or `Mon DD,YY` (SHLP reports)   |
| Negative values  | Plain integer (`-3`), no parentheses                 |
| Boolean fields   | `Y`/`N` or `1`/`0`                                  |
| Line endings     | CRLF (Windows)                                       |

#### Inventory Columns — Native Export (Utilities > Export)

| Paladin Column     | → Canonical Field     | Notes                                   |
|--------------------|-----------------------|-----------------------------------------|
| `partnumber`       | `sku`                 | Primary identifier, one word no spaces  |
| `altpartnumber`    | `sku` (alt)           | Alternate part number                   |
| `mfgpartnumber`    | `sku` (mfg)           | Manufacturer part number                |
| `description1`     | `description`         | Primary description (30-40 chars)       |
| `description2`     | `description` (ext)   | Extended description                    |
| `stockonhand`      | `quantity`            | Current QOH, single word                |
| `mktcost`          | `cost`                | Market cost (last vendor invoice cost)  |
| `avgcost`          | `cost` (avg)          | Weighted average cost                   |
| `netprice`         | `revenue`             | Current selling price                   |
| `plvl_price1`      | `revenue`             | Price Level 1 (retail/regular)          |
| `plvl_price2`      | `revenue` (tier 2)    | Price Level 2 (contractor/pro)          |
| `plvl_price3`      | `revenue` (tier 3)    | Price Level 3 (wholesale)               |
| `supplier_number1` | `vendor`              | Primary vendor number                   |
| `supplier_number2` | `vendor` (alt)        | Secondary vendor number                 |
| `deptid`           | `category`            | Department ID                           |
| `classid`          | `category` (sub)      | Class ID                                |
| `fineline`         | `category` (fine)     | Fineline code                           |
| `minorderqty`      | `reorder_point`       | Min order / reorder point               |
| `maxorderqty`      | `reorder_point` (max) | Max order / reorder-up-to level         |
| `upccode`          | `sku` (barcode)       | UPC barcode                             |
| `binlocation`      | `location`            | Bin/shelf location                      |
| `onorderqty`       | (extended)            | Quantity on order                       |
| `lastsaledate`     | `last_sale_date`      | Format: YYYYMMDD                        |
| `lastreceiveddate` | `last_purchase_date`  | Format: YYYYMMDD                        |
| `qtysold_ytd`      | `sold`                | YTD units sold                          |
| `qtysold_mtd`      | `sold` (mtd)          | MTD units sold                          |
| `status`           | `status`              | A=Active, I=Inactive, D=Discontinued   |
| `margin`           | `margin`              | ⚠ UNRELIABLE — always recalculate      |
| `taxable`          | `tax`                 | Y/N                                     |

#### Inventory Columns — Report-Style Export (Reports > Inventory)

| Paladin Column        | → Canonical Field     | Notes                              |
|-----------------------|-----------------------|------------------------------------|
| `SKU`                 | `sku`                 | Same as `partnumber`               |
| `Vendor SKU`          | (extended)            | Vendor catalog number              |
| `Vendor`              | `vendor`              | Vendor name (not number)           |
| `Cat.`                | `category`            | Category                           |
| `Dpt.`                | `category` (dept)     | Department                         |
| `BIN`                 | `location`            | Bin location                       |
| `Description Full`    | `description`         | Combined desc1+desc2               |
| `Description Short`   | `description`         | Short description only             |
| `Qty.`                | `quantity`            | ⚠ Includes negatives (preferred)  |
| `Qty On Hand`         | `quantity` (phys)     | Physical count, never negative     |
| `Std. Cost`           | `cost`                | Standard / market cost             |
| `Avg. Cost`           | `cost` (avg)          | Weighted average cost              |
| `Retail`              | `revenue`             | Selling price                      |
| `Sug. Retail`         | `revenue` (msrp)      | Suggested retail / MSRP            |
| `Margin @Cost`        | `margin`              | ⚠ UNRELIABLE                      |
| `Inventory @Cost`     | `sub_total`           | qty × cost                         |
| `Inventory @Retail`   | `sub_total` (retail)  | qty × retail                       |
| `Inventory @AvgCost`  | `sub_total` (avg)     | qty × avg cost                     |
| `Sales`               | `sold`                | Units sold (period)                |
| `$ Sold`              | (extended)            | Dollar sales                       |
| `Last Sale`           | `last_sale_date`      | YYYYMMDD                           |
| `Last Purchase`       | `last_purchase_date`  | YYYYMMDD                           |
| `On Order`            | (extended)            | Qty on order                       |
| `Min.`                | `reorder_point`       | Reorder point                      |
| `Max.`                | `reorder_point` (max) | Reorder-up-to                      |
| `Pkg.`                | `package_qty`         | Package quantity                   |
| `Barcode`             | `sku` (barcode)       | UPC barcode                        |
| `Returns`             | `return_flag`         | Return qty                         |
| `Retail Dif.`         | (extended)            | Retail vs suggested difference     |

---

### 2.2 Epicor Eagle

**Vendor:** Epicor Software (formerly Activant)
**Market:** Hardware, lumber, automotive, industrial distribution
**Confidence:** HIGH (domain knowledge + codebase references)

#### Format Details

| Property         | Value                                                |
|------------------|------------------------------------------------------|
| File format      | Pipe-delimited (`|`) for DAFL, CSV via Report Builder|
| Encoding         | UTF-8 or Windows-1252                                |
| Header rows      | 1 (DAFL may omit headers)                            |
| Date format      | `MM/DD/YY` (Import Designer), `YYYY-MM-DD` (exports) |
| Commas in data   | Converted to `?` during DAFL import                  |

#### Inventory Columns (Item Master / DAFL)

| Eagle Column         | → Canonical Field     | Notes                              |
|----------------------|-----------------------|------------------------------------|
| `ean`                | `sku`                 | European Article Number / item ID  |
| `upc`                | `sku` (barcode)       | UPC barcode                        |
| `sku`                | `sku`                 | Report Builder SKU                 |
| `alt_sku`            | `sku` (alt)           | Alternate SKU                      |
| `mfg_no`             | `sku` (mfg)           | Manufacturer item number           |
| `vendor_no`          | `vendor`              | Primary vendor number              |
| `vendor_item_no`     | (extended)            | Vendor's item number               |
| `desc`               | `description`         | Primary description (30 char)      |
| `desc2`              | `description` (ext)   | Secondary description              |
| `long_desc`          | `description` (long)  | Extended description               |
| `qty_oh`             | `quantity`            | Quantity on hand                   |
| `qty_bo`             | (extended)            | Quantity on backorder              |
| `qty_comm`           | (extended)            | Quantity committed                 |
| `qty_avl`            | `quantity` (avail)    | Quantity available (oh - committed)|
| `qty_on_ord`         | (extended)            | Quantity on order                  |
| `min_qty`            | `reorder_point`       | Minimum stock / reorder point      |
| `max_qty`            | `reorder_point` (max) | Maximum stock level                |
| `reorder_qty`        | (extended)            | Reorder quantity                   |
| `posting_cost`       | `cost`                | Current posting cost (for COGS)    |
| `avg_cost`           | `cost` (avg)          | Average cost                       |
| `last_cost`          | `cost` (last)         | Last cost paid                     |
| `replacement_cost`   | `cost` (repl)         | Replacement cost                   |
| `landed_cost`        | `cost` (landed)       | Landed cost with freight           |
| `core_cost`          | (extended)            | Core charge (auto/plumbing)        |
| `retail`             | `revenue`             | Current retail price               |
| `price_1`–`price_4`  | `revenue` (tiers)     | Price levels 1-4                   |
| `sug_retail`         | `revenue` (msrp)      | Suggested retail                   |
| `msrp`               | `revenue` (msrp)      | Manufacturer suggested retail      |
| `promo_price`        | `revenue` (promo)     | Promotional price                  |
| `dept`               | `category`            | Department code                    |
| `class`              | `category` (class)    | Classification code                |
| `fineline`           | `category` (fine)     | Fineline class code                |
| `sub_dept`           | `category` (sub)      | Sub-department                     |
| `ytd_sold`           | `sold`                | Year-to-date units sold            |
| `ly_sold`            | `sold` (ly)           | Last year units sold               |
| `mtd_sold`           | `sold` (mtd)          | Month-to-date sold                 |
| `wtd_sold`           | `sold` (wtd)          | Week-to-date sold                  |
| `last_sold`          | `last_sale_date`      | Date of last sale                  |
| `last_rcpt`          | `last_purchase_date`  | Date of last receipt               |
| `last_count`         | (extended)            | Date of last physical count        |
| `turns`              | `sold` (turns)        | Turn rate                          |
| `gmroi`              | (extended)            | Gross Margin ROI                   |
| `margin`             | `margin`              | Margin percentage                  |
| `status`             | `status`              | A=Active, D=Discontinued           |
| `pkg_qty`            | `package_qty`         | Package quantity                   |
| `weight`             | (extended)            | Item weight                        |
| `uom`                | `package_qty` (uom)   | Unit of measure (EA, CS, etc.)     |
| `bin_loc`            | `location`            | Bin location                       |
| `aisle`              | `location` (aisle)    | Aisle location                     |

---

### 2.3 Epicor BisTrack

**Vendor:** Epicor Software
**Market:** Lumber yards, building materials distribution
**Confidence:** MEDIUM-HIGH (domain knowledge, limited public docs)

#### Lumber-Specific Measurement Fields (unique to BisTrack)

| BisTrack Column       | → Canonical Field      | Notes                             |
|-----------------------|------------------------|-----------------------------------|
| `board_feet` / `bf`   | (extended: lumber_qty) | Standard dimensional lumber unit  |
| `linear_feet` / `lf`  | (extended: lumber_qty) | Molding, trim, decking            |
| `square_feet` / `sf`  | (extended: lumber_qty) | Sheet goods, panels, flooring     |
| `tally`               | (extended)             | Hardwood piece-by-piece tracking  |
| `mbf`                 | (extended)             | Thousand board feet (wholesale)   |
| `pieces`              | `quantity`             | Piece count                       |
| `thickness`           | (extended)             | Nominal thickness                 |
| `width`               | (extended)             | Nominal width                     |
| `length`              | (extended)             | Standard length                   |
| `grade`               | (extended)             | Lumber grade (Select, #1, #2)     |
| `species`             | (extended)             | Wood species (SPF, SYP, Doug Fir) |
| `treatment`           | (extended)             | PT, KD, Green                     |
| `moisture_content`    | (extended)             | Moisture percentage               |
| `selling_uom`         | `package_qty` (uom)    | Selling unit (EA, BF, LF, MBF)   |
| `buying_uom`          | (extended)             | Purchasing unit                   |
| `pricing_uom`         | (extended)             | Pricing unit                      |
| `stocking_uom`        | (extended)             | Inventory unit                    |
| `cost_per_bf`         | `cost` (per bf)        | Cost per board foot               |
| `cost_per_lf`         | (extended)             | Cost per linear foot              |
| `cost_per_mbf`        | (extended)             | Cost per thousand board feet      |
| `yard_location`       | `location`             | Yard location (lumber yards)      |
| `rack`                | `location` (rack)      | Rack location                     |
| `bay`                 | `location` (bay)       | Bay number                        |
| `nominal_size`        | (extended)             | e.g., "2x4", "2x6"               |
| `surfacing`           | (extended)             | S4S, Rough, etc.                  |
| `drying`              | (extended)             | KD, AD, Green                     |
| `certification`       | (extended)             | FSC, SFI, etc.                    |
| `bundle_qty`          | `package_qty`          | Pieces per bundle                 |
| `lift_qty`            | `package_qty` (lift)   | Pieces per lift                   |
| `random_length`       | (extended)             | Random length flag (Y/N)          |

---

### 2.4 Spruce POS

**Vendor:** ECI Software Solutions (formerly Spruce Software)
**Market:** Lumber yards, building materials, hardware
**Confidence:** LOW (closed ecosystem, very limited public documentation)

#### Likely Inventory Columns

| Spruce Column          | → Canonical Field     | Notes                            |
|------------------------|-----------------------|----------------------------------|
| `Item Number` / `SKU`  | `sku`                | Primary identifier               |
| `Description`          | `description`        | Item description                 |
| `Department`           | `category`           | Department                       |
| `Category`             | `category` (sub)     | Category                         |
| `Vendor`               | `vendor`             | Vendor name                      |
| `Vendor Item Number`   | (extended)           | Vendor part number               |
| `On Hand Qty`          | `quantity`           | Quantity on hand                 |
| `Cost`                 | `cost`               | Unit cost                        |
| `Price` / `Retail`     | `revenue`            | Retail price                     |
| `Unit of Measure`      | `package_qty` (uom)  | EA, BF, LF, MBF, SqFt, Bundle   |
| `Board Feet` / `BdFt`  | (extended)           | Lumber-specific                  |
| `Linear Feet` / `LinFt`| (extended)           | Lumber-specific                  |
| `Species`              | (extended)           | Wood species                     |
| `Grade`                | (extended)           | Lumber grade                     |
| `Reorder Point`        | `reorder_point`      | Min stock level                  |
| `Location`             | `location`           | Store/yard location              |
| `Last Received Date`   | `last_purchase_date` | Last receipt                     |
| `Last Sold Date`       | `last_sale_date`     | Last sale                        |

> ⚠ **Action required:** Obtain a sample export from a Spruce installation to
> verify exact column headers. Spruce uses Pervasive SQL (Btrieve) backend.

---

### 2.5 NCR Counterpoint

**Vendor:** NCR Corporation
**Market:** Specialty retail, hardware, apparel, footwear
**Confidence:** VERY HIGH (SQL Server schema well-documented)

#### Format Details

| Property         | Value                                                |
|------------------|------------------------------------------------------|
| File format      | Tab-delimited or CSV                                 |
| Encoding         | UTF-8                                                |
| Date format      | `YYYY-MM-DD` (SQL Server) or `MM/DD/YYYY` (reports)  |
| Database         | SQL Server (`cpsql` database)                        |

#### Inventory Columns (IM_ITEM table)

| Counterpoint Column  | → Canonical Field     | Notes                              |
|----------------------|-----------------------|------------------------------------|
| `item_no`            | `sku`                 | Primary key                        |
| `descr`              | `description`         | Main description                   |
| `long_descr`         | `description` (long)  | Long description                   |
| `short_descr`        | `description` (short) | Short description                  |
| `addl_descr_1`       | `description` (add1)  | Additional description 1           |
| `addl_descr_2`       | `description` (add2)  | Additional description 2           |
| `barcod`             | `sku` (barcode)       | Barcode / UPC                      |
| `upc_cod`            | `sku` (upc)           | UPC code                           |
| `item_typ`           | (extended)            | I=Inventory, N=Non, S=Serialized   |
| `categ_cod`          | `category`            | Category code                      |
| `subcat_cod`         | `category` (sub)      | Subcategory code                   |
| `acct_cod`           | (extended)            | Account code                       |
| `attr_cod_1`–`3`     | (extended)            | Attribute codes                    |
| `qty_on_hnd`         | `quantity`            | Quantity on hand                   |
| `qty_avail`          | `quantity` (avail)    | Quantity available                 |
| `net_qty`            | `quantity` (net)      | Net quantity                       |
| `qty_on_ord`         | (extended)            | Quantity on order                  |
| `qty_on_po`          | (extended)            | Quantity on PO                     |
| `qty_on_bo`          | (extended)            | Quantity backordered               |
| `qty_commit`         | (extended)            | Quantity committed                 |
| `min_qty`            | `reorder_point`       | Minimum qty / reorder point        |
| `max_qty`            | `reorder_point` (max) | Maximum qty                        |
| `reord_qty`          | (extended)            | Reorder quantity                   |
| `prc_1`              | `revenue`             | Price level 1 (retail)             |
| `prc_2`–`prc_10`     | `revenue` (tiers)     | Price levels 2-10                  |
| `reg_prc`            | `revenue` (reg)       | Regular price                      |
| `min_prc`            | (extended)            | Minimum price                      |
| `min_mrgn_pct`       | `margin` (min)        | Minimum margin %                   |
| `avg_cost`           | `cost` (avg)          | Average cost                       |
| `lst_cost`           | `cost`                | Last cost (most recent)            |
| `std_cost`           | `cost` (std)          | Standard cost                      |
| `orig_cost`          | `cost` (orig)         | Original cost                      |
| `prev_cost`          | `cost` (prev)         | Previous cost                      |
| `vend_no`            | `vendor`              | Primary vendor number              |
| `item_vend_no`       | `vendor` (item)       | Item-vendor number                 |
| `vend_part_no`       | (extended)            | Vendor part number                 |
| `lst_recv_dat`       | `last_purchase_date`  | Last received date                 |
| `lst_sold_dat`       | `last_sale_date`      | Last sold date                     |
| `lst_ord_dat`        | (extended)            | Last ordered date                  |
| `lst_cnt_dat`        | (extended)            | Last count date                    |
| `fst_recv_dat`       | (extended)            | First received date                |
| `lst_maint_dt`       | `last_sale_date` (alt)| Last maintenance date              |
| `stat`               | `status`              | Status (A=Active, etc.)            |
| `sell_unit`          | `package_qty` (uom)   | Selling unit of measure            |
| `stk_unit`           | `package_qty` (stk)   | Stock unit                         |
| `loc_id`             | `location`            | Location ID                        |
| `is_ecomm_item`      | (extended)            | E-commerce flag                    |
| `is_txbl`            | `tax`                 | Taxable flag                       |
| `is_weighed`         | (extended)            | Weighed item flag                  |
| `weight`             | (extended)            | Item weight                        |

---

### 2.6 POSIM

**Vendor:** POSIM (Point of Sale and Inventory Management)
**Market:** Specialty retail, hardware
**Confidence:** LOW-MEDIUM (limited public documentation)

#### Physical Inventory Import Format (confirmed from docs)

| Column | Position | Required | Notes                       |
|--------|----------|----------|-----------------------------|
| `SKU`  | A        | Yes      | Must preserve leading zeros |
| `Quantity` | B    | No       | Physical count              |
| `Location` | C    | No       | Location code               |
| `Who`  | D        | No       | Counter initials            |

#### Likely Inventory Export Columns

| POSIM Column          | → Canonical Field     | Notes                     |
|-----------------------|-----------------------|---------------------------|
| `Item Number`/`ItemNum`| `sku`               | Primary identifier        |
| `Description`         | `description`         | Item description          |
| `Department`          | `category`            | Department                |
| `Category`            | `category` (sub)      | Category                  |
| `Vendor`              | `vendor`              | Vendor name               |
| `UPC` / `Barcode`     | `sku` (barcode)       | UPC code                  |
| `Quantity on Hand`    | `quantity`            | QOH                       |
| `Cost`                | `cost`                | Unit cost                 |
| `Average Cost`        | `cost` (avg)          | Average cost              |
| `MSRP`                | `revenue` (msrp)      | Suggested retail          |
| `Price 1`             | `revenue`             | Primary retail price      |
| `Price 2`             | `revenue` (tier 2)    | Secondary price level     |
| `Minimum Stock`       | `reorder_point`       | Min stock                 |
| `Maximum Stock`       | `reorder_point` (max) | Max stock                 |
| `Last Received`       | `last_purchase_date`  | Last receipt date         |
| `Last Sold`           | `last_sale_date`      | Last sale date            |
| `Location`            | `location`            | Location code             |
| `Status`              | `status`              | Active/Inactive           |

> ⚠ **Action required:** Obtain a sample export from a POSIM installation.
> POSIM uses FileMaker-based backend. Text formatting required to preserve
> leading zeros in SKU field.

---

### 2.7 IdoSoft POS

**Vendor:** IdoSoft
**Market:** Independent hardware stores (smaller operations)
**Confidence:** HIGH (sample store adapter fully implemented, real data validated)

#### Format Details

| Property         | Value                                                |
|------------------|------------------------------------------------------|
| File format      | CSV (comma-delimited)                                |
| Encoding         | UTF-8 with `errors="replace"`                        |
| Date format      | `YYYYMMDD` (custom_1) or `Mon DD,YY` (SHLP)         |
| Row count        | Can exceed 150K+ rows per store                      |

#### Inventory Columns — custom_1.csv Format

| IdoSoft Column         | → Canonical Field     | Notes                           |
|------------------------|-----------------------|---------------------------------|
| `SKU`                  | `sku`                 | Primary identifier              |
| `Vendor SKU`           | (extended)            | Vendor catalog number           |
| `Vendor`               | `vendor`              | Vendor name                     |
| `Alt. Vendor/Mfgr.`   | `vendor` (alt)        | Alternate vendor/manufacturer   |
| `Cat.`                 | `category`            | Category                        |
| `Dpt.`                 | `category` (dept)     | Department                      |
| `BIN`                  | `location`            | Bin location                    |
| `Description Full`     | `description`         | Full description                |
| `Description Short`    | `description` (short) | Short description               |
| `Qty.`                 | `quantity`            | ⚠ Includes negatives (3,958 in sample store) |
| `Qty On Hand`          | `quantity` (phys)     | Physical count, never negative  |
| `Std. Cost`            | `cost`                | Standard cost                   |
| `Avg. Cost`            | `cost` (avg)          | Average cost (preferred)        |
| `Retail`               | `revenue`             | Retail price                    |
| `Sug. Retail`          | `revenue` (msrp)      | Suggested retail                |
| `Margin @Cost`         | `margin`              | ⚠ May be unreliable            |
| `Inventory @Cost`      | `sub_total`           | qty × cost                      |
| `Inventory @AvgCost`   | `sub_total` (avg)     | qty × avg cost                  |
| `Inventory @Retail`    | `sub_total` (retail)  | qty × retail                    |
| `Sales`                | `sold`                | Units sold                      |
| `$ Sold`               | (extended)            | Dollar sales (has commas)       |
| `Last Sale`            | `last_sale_date`      | YYYYMMDD format                 |
| `Last Purchase`        | `last_purchase_date`  | YYYYMMDD format                 |
| `On Order`             | (extended)            | Qty on order                    |
| `Min.`                 | `reorder_point`       | Reorder point                   |
| `Max.`                 | `reorder_point` (max) | Reorder-up-to                   |
| `Pkg.`                 | `package_qty`         | Package quantity                |
| `Barcode`              | `sku` (barcode)       | UPC barcode                     |
| `Mfgr. SKU`            | `sku` (mfg)           | Manufacturer SKU                |
| `Returns`              | `return_flag`         | Return quantity                 |

#### SHLP / YTD Report Columns

| Column            | → Canonical Field     | Notes                          |
|-------------------|-----------------------|--------------------------------|
| `SKU`             | `sku`                 |                                |
| `Description`     | `description`         |                                |
| `Vendor`          | `vendor`              |                                |
| `Vendor SKU`      | (extended)            |                                |
| `Stock`           | `quantity`            | Uses "Stock" not "Qty"         |
| `Avg. Cost`       | `cost`                |                                |
| `Gross Sales`     | (extended)            | Dollar sales                   |
| `Last Sale`       | `last_sale_date`      | `Mon DD,YY` format             |
| `Real Date`       | `last_purchase_date`  | YYYYMMDD                       |
| `Year Total`      | `sold`                | Annual total                   |
| `Jan`–`Dec`       | (extended)            | Monthly sales columns          |

#### IdoSoft Detection Signatures

The key fingerprint for IdoSoft is the combination of:
- `wholesale` as cost field name (in test schemas)
- `qty` as quantity field name (simple, short)
- `Qty.` with trailing period
- `Margin @Cost` notation
- `Inventory @Cost` / `@Retail` / `@AvgCost` notation

---

## 3. Tier 2: General Retail POS Systems

### 3.1 Lightspeed R-Series

**Vendor:** Lightspeed Commerce
**Market:** Multi-vertical retail (hardware, apparel, bike, outdoor)
**Confidence:** VERY HIGH (already in codebase + extensive API docs)

| Lightspeed R Column        | → Canonical Field     | Notes                     |
|----------------------------|-----------------------|---------------------------|
| `System ID`                | `sku`                 | Internal Lightspeed ID    |
| `Custom SKU`               | `sku` (custom)        | User-defined SKU          |
| `Manufacturer SKU`         | `sku` (mfg)           | Manufacturer SKU          |
| `Item/Description`         | `description`         | Item name                 |
| `UPC`                      | `sku` (upc)           | UPC code                  |
| `EAN`                      | `sku` (ean)           | EAN code                  |
| `Default Cost`             | `cost`                | Default cost              |
| `Default Price`            | `revenue`             | Default retail price      |
| `MSRP`                     | `revenue` (msrp)      | Suggested retail          |
| `Vendor`                   | `vendor`              | Vendor name               |
| `Vendor ID`                | `vendor` (id)         | Vendor ID                 |
| `Vendor Cost`              | `cost` (vendor)       | Vendor-specific cost      |
| `Brand`                    | `vendor` (brand)      | Brand name                |
| `Category`                 | `category`            | Category                  |
| `Sub Category` (x3)        | `category` (sub)      | Up to 3 sub-category levels|
| `Tax Class`                | `tax`                 | Tax class                 |
| `Discountable`             | `discount`            | Boolean                   |
| `[STORE] - QOH`            | `quantity`            | QOH per location          |
| `[STORE] - Reorder Point`  | `reorder_point`       | Per location              |
| `[STORE] - Reorder Level`  | (extended)            | Reorder-up-to per location|

**API field names (lowercase underscore):**
- `system_id`, `custom_sku`, `manufacturer_sku`, `default_cost`, `default_price`
- `variant_inventory_qty`, `stock_min`, `vendor_id`, `per_item_supply_price`

---

### 3.2 Lightspeed X-Series (Vend)

| X-Series / Vend Column  | → Canonical Field     | Notes                       |
|--------------------------|-----------------------|-----------------------------|
| `product_handle`         | `sku`                 | X-Series primary ID         |
| `product_name`           | `description`         | Product name                |
| `variant_name`           | `description` (var)   | Variant name                |
| `variant_sku`            | `sku` (variant)       | Variant SKU                 |
| `supply_price`           | `cost`                | Cost / supply price         |
| `retail_price`           | `revenue`             | Retail price                |
| `markup`                 | `margin`              | Markup percentage           |
| `brand_name`             | `vendor` (brand)      | Brand                       |
| `supplier_name`          | `vendor`              | Supplier name               |
| `supplier_code`          | `vendor` (code)       | Supplier code               |
| `product_type`           | `category`            | Category / type             |
| `tags`                   | `category` (tags)     | Tags                        |
| `outlet_name`            | `location`            | Store / outlet              |
| `current_inventory`      | `quantity`            | QOH                         |
| `reorder_point`          | `reorder_point`       | Reorder point               |
| `reorder_amount`         | (extended)            | Reorder quantity            |
| `variant_loyalty_value`  | (extended)            | Loyalty points value        |
| `product_active`         | `status`              | Active flag                 |

---

### 3.3 Retail Pro

**Vendor:** Retail Pro International
**Market:** Specialty retail, fashion, footwear
**Confidence:** HIGH (well-documented data dictionary)

| Retail Pro Column    | → Canonical Field     | Notes                          |
|----------------------|-----------------------|--------------------------------|
| `SID`                | `sku`                 | System ID (unique, internal)   |
| `UPC`                | `sku` (upc)           | UPC barcode                    |
| `ALU`                | `sku` (alt)           | Alternate Lookup (user SKU)    |
| `Item Lookup Code`   | `sku` (lookup)        | Item lookup code               |
| `Description 1`      | `description`         | Primary description            |
| `Description 2`      | `description` (ext)   | Secondary description          |
| `DCS`                | `category`            | Dept/Class/Subclass composite  |
| `Department Name`    | `category` (dept)     | Department name                |
| `Class Name`         | `category` (class)    | Class name                     |
| `Vendor Code`        | `vendor`              | Vendor code                    |
| `Vendor Name`        | `vendor` (name)       | Vendor name                    |
| `Vendor Part Number` | (extended)            | Vendor part number             |
| `Cost`               | `cost`                | Unit cost                      |
| `Last Cost`          | `cost` (last)         | Last cost paid                 |
| `Average Cost`       | `cost` (avg)          | Average cost                   |
| `Price 1`            | `revenue`             | Primary retail price           |
| `Price 2`–`Price 5`  | `revenue` (tiers)     | Additional price levels        |
| `Qty OH`             | `quantity`            | Quantity on hand               |
| `Qty Avail`          | `quantity` (avail)    | Quantity available             |
| `Qty OO`             | (extended)            | Quantity on order              |
| `Qty Committed`      | (extended)            | Quantity committed             |
| `Min`                | `reorder_point`       | Minimum stock                  |
| `Max`                | `reorder_point` (max) | Maximum stock                  |
| `Last Sold`          | `last_sale_date`      | Last sold date                 |
| `Last Received`      | `last_purchase_date`  | Last received date             |
| `Margin`             | `margin`              | Margin percentage              |
| `Store Code`         | `location`            | Store code                     |
| `Subsidiary`         | `location` (sub)      | Subsidiary                     |
| `Season Code`        | (extended)            | Season classification          |
| `Inventory Status`   | `status`              | Active/Inactive/etc.           |

---

### 3.4 KORONA POS

**Vendor:** COMBASE AG
**Market:** Multi-vertical retail, hardware, specialty
**Confidence:** MEDIUM-HIGH (REST API documented)

| KORONA Column              | → Canonical Field     | Notes                       |
|----------------------------|-----------------------|-----------------------------|
| `Number`                   | `sku`                 | Product number              |
| `Name`                     | `description`         | Product name                |
| `Product Code`             | `sku` (code)          | Product code                |
| `Commodity Group`          | `category`            | ⚠ KORONA-unique term       |
| `Commodity Group Number`   | `category` (code)     | Category code               |
| `Supplier`                 | `vendor`              | Vendor / supplier           |
| `Supplier Product Number`  | (extended)            | Vendor part number          |
| `Sector`                   | `category` (sector)   | KORONA organizational level |
| `Assortment`               | `category` (assort)   | KORONA assortment group     |
| `Purchase Price`           | `cost`                | Unit cost                   |
| `Retail Price`             | `revenue`             | Retail price                |
| `Sales Price`              | `revenue` (sale)      | Current sales price         |
| `Special Price`            | `revenue` (promo)     | Promotional price           |
| `Current Stock`            | `quantity`            | QOH                         |
| `Min Stock`                | `reorder_point`       | Minimum stock               |
| `Max Stock`                | `reorder_point` (max) | Maximum stock               |
| `Optimal Stock`            | `reorder_point` (opt) | ⚠ KORONA-unique            |
| `Reorder Level`            | `reorder_point` (alt) | Reorder trigger             |
| `Sold Quantity`            | `sold`                | Units sold                  |
| `Revenue`                  | (extended)            | Revenue amount              |
| `Profit`                   | `margin` (dollars)    | Profit amount               |
| `Margin`                   | `margin`              | Margin percentage           |
| `Barcode` / `EAN`          | `sku` (barcode)       | Barcode                     |
| `Organizational Unit`      | `location`            | ⚠ KORONA-unique for store  |
| `Warehouse`                | `location` (wh)       | Warehouse                   |
| `Inventory Difference`     | `qty_difference`      | Variance                    |
| `Active`                   | `status`              | Boolean                     |

---

### 3.5 Rain POS

**Vendor:** Rain Retail Software
**Market:** Bike shops, music stores, outdoor/ski shops
**Confidence:** MEDIUM

| Rain POS Column        | → Canonical Field     | Notes                        |
|------------------------|-----------------------|------------------------------|
| `Product ID`           | `sku`                 | Internal ID                  |
| `SKU`                  | `sku` (display)       | Display SKU                  |
| `Barcode`              | `sku` (barcode)       | Barcode                      |
| `Product Name`         | `description`         | Product name                 |
| `Brand`                | `vendor` (brand)      | Brand (separate from vendor) |
| `Category`             | `category`            | Category                     |
| `Subcategory`          | `category` (sub)      | Subcategory                  |
| `Vendor`               | `vendor`              | Vendor                       |
| `Vendor SKU`           | (extended)            | Vendor's SKU                 |
| `Cost`                 | `cost`                | Unit cost                    |
| `MSRP`                 | `revenue` (msrp)      | Manufacturer SRP             |
| `Retail Price`         | `revenue`             | Retail price                 |
| `Sale Price`           | `revenue` (sale)      | Sale price                   |
| `Online Price`         | `revenue` (online)    | Webstore price               |
| `Quantity On Hand`     | `quantity`            | QOH                          |
| `Quantity Available`   | `quantity` (avail)    | Available qty                |
| `Quantity On Order`    | (extended)            | On order qty                 |
| `Reorder Point`        | `reorder_point`       | Reorder trigger              |
| `Location`             | `location`            | Store location               |
| `Last Sold`            | `last_sale_date`      | Last sale date               |
| `Last Received`        | `last_purchase_date`  | Last receipt date            |
| `Catalog ID`           | (extended)            | ⚠ Rain-unique catalog ref   |
| `Serialized`           | (extended)            | Serial tracking flag         |
| `Year`                 | (extended)            | Model year (bikes)           |
| `Season`               | (extended)            | Season code                  |
| `Gender`               | (extended)            | Gender attribute             |
| `Active`               | `status`              | Active flag                  |

---

### 3.6 MicroBiz

**Vendor:** MicroBiz Inc.
**Market:** Small-medium retail, hardware, specialty
**Confidence:** MEDIUM

| MicroBiz Column        | → Canonical Field     | Notes                        |
|------------------------|-----------------------|------------------------------|
| `Item Number`          | `sku`                 | Primary SKU                  |
| `Item Name`            | `description`         | Item name                    |
| `Description`          | `description` (long)  | Full description             |
| `Long Description`     | `description` (ext)   | Extended description         |
| `Department`           | `category`            | Department                   |
| `Category`             | `category` (sub)      | Category                     |
| `Vendor`               | `vendor`              | Vendor                       |
| `Vendor Item Number`   | (extended)            | Vendor part number           |
| `Manufacturer`         | `vendor` (mfg)        | Manufacturer                 |
| `UPC`                  | `sku` (upc)           | UPC barcode                  |
| `Cost`                 | `cost`                | Unit cost                    |
| `Average Cost`         | `cost` (avg)          | Average cost                 |
| `Last Cost`            | `cost` (last)         | Last cost paid               |
| `Price`                | `revenue`             | Retail price                 |
| `MSRP`                 | `revenue` (msrp)      | Suggested retail             |
| `Compare At Price`     | `revenue` (compare)   | ⚠ MicroBiz-unique           |
| `Quantity On Hand`     | `quantity`            | QOH                          |
| `Quantity Available`   | `quantity` (avail)    | Available qty                |
| `Quantity On Order`    | (extended)            | On order qty                 |
| `Quantity Reserved`    | (extended)            | ⚠ MicroBiz-unique           |
| `Reorder Point`        | `reorder_point`       | Min stock                    |
| `Bin Location`         | `location`            | Bin / shelf location         |
| `Last Sale Date`       | `last_sale_date`      | Last sale                    |
| `Last Receipt Date`    | `last_purchase_date`  | Last receipt                 |
| `Track Inventory`      | (extended)            | ⚠ MicroBiz boolean flag     |
| `Matrix`               | (extended)            | ⚠ Matrix/variant indicator   |
| `eCommerce`            | (extended)            | ⚠ E-commerce sync flag      |
| `Active`               | `status`              | Yes/No                       |

---

### 3.7 Celerant Command Retail

**Vendor:** Celerant Technology
**Market:** Multi-vertical retail
**Confidence:** MEDIUM (closed ecosystem)

| Celerant Column        | → Canonical Field     | Notes                        |
|------------------------|-----------------------|------------------------------|
| `Item Number`          | `sku`                 | Primary SKU                  |
| `Alt Item Number`      | `sku` (alt)           | Alternate SKU                |
| `UPC`                  | `sku` (upc)           | UPC barcode                  |
| `Description`          | `description`         | Item description             |
| `Description 2`       | `description` (ext)   | Extended description         |
| `Department`           | `category`            | Department                   |
| `Class`                | `category` (class)    | Class                        |
| `Subclass`             | `category` (sub)      | Subclass                     |
| `Vendor Code`          | `vendor`              | Vendor code                  |
| `Vendor Name`          | `vendor` (name)       | Vendor name                  |
| `Vendor Item Number`   | (extended)            | Vendor part number           |
| `Cost`                 | `cost`                | Unit cost                    |
| `Average Cost`         | `cost` (avg)          | Average cost                 |
| `Last Cost`            | `cost` (last)         | Last cost paid               |
| `Price`                | `revenue`             | Retail price                 |
| `MSRP`                 | `revenue` (msrp)      | Suggested retail             |
| `Qty On Hand`          | `quantity`            | QOH                          |
| `Qty Available`        | `quantity` (avail)    | Available qty                |
| `Qty On Order`         | (extended)            | On order qty                 |
| `Qty Committed`        | (extended)            | Committed qty                |
| `Reorder Point`        | `reorder_point`       | Reorder trigger              |
| `Last Sold Date`       | `last_sale_date`      | Last sale                    |
| `Last Received Date`   | `last_purchase_date`  | Last receipt                 |
| `Margin %`             | `margin`              | Margin percentage            |
| `Location`             | `location`            | Store location               |
| `Bin`                  | `location` (bin)      | Bin location                 |
| `Status`               | `status`              | Active/Inactive/Discontinued |

---

### 3.8 Shopify POS

**Already in `standard_fields.yaml`.** Key mappings:

| Shopify Column             | → Canonical Field     |
|----------------------------|-----------------------|
| `handle`                   | `sku`                 |
| `variant_sku`              | `sku` (variant)       |
| `variant_barcode`          | `sku` (barcode)       |
| `title` / `body_html`     | `description`         |
| `on_hand_current`          | `quantity`            |
| `cost_per_item`            | `cost`                |
| `variant_price`            | `revenue`             |
| `variant_compare_at_price` | `revenue` (msrp)      |
| `type` / `tags`            | `category`            |

---

### 3.9 Square POS

**Already in `standard_fields.yaml`.** Key mappings:

| Square Column              | → Canonical Field     |
|----------------------------|-----------------------|
| `token`                    | `sku`                 |
| `reference_handle`         | `sku` (ref)           |
| `gtin`                     | `sku` (barcode)       |
| `new_quantity`             | `quantity`            |
| `stock_alert_count`        | `reorder_point`       |
| `default_vendor_name`      | `vendor`              |
| `reporting_category`       | `category`            |

---

### 3.10 Clover POS

**Already in `standard_fields.yaml`.** Key mappings:

| Clover Column    | → Canonical Field     |
|------------------|-----------------------|
| `clover_id`      | `sku`                 |
| `labels`         | `category`            |

---

### 3.11 Microsoft Dynamics RMS

**Already in `standard_fields.yaml`.** Key mappings:

| Dynamics RMS Column    | → Canonical Field     |
|------------------------|-----------------------|
| `itemlookupcode`       | `sku`                 |
| `itemdescription`      | `description`         |
| `itemsubdescription2`  | `description` (ext)   |
| `pricea`               | `revenue`             |
| `priceb`               | `revenue` (tier 2)    |
| `pricec`               | `revenue` (tier 3)    |

---

## 4. Tier 3: Co-op / Distributor Systems

### 4.1 Do It Best

**Vendor:** Do It Best Corp (Fort Wayne, IN)
**System:** best systems / TransActPOS
**Confidence:** MEDIUM (member portal, limited public docs)

| Do It Best Column      | → Canonical Field     | Notes                          |
|------------------------|-----------------------|--------------------------------|
| `Member_Number`        | (extended)            | 6-digit member ID              |
| `Item_Number`          | `sku`                 | 7-digit Do It Best catalog #   |
| `UPC`                  | `sku` (upc)           | UPC-A barcode (12 digits)      |
| `Description`          | `description`         | Up to 40 characters            |
| `Vendor_Name`          | `vendor`              | Supplier / manufacturer        |
| `Vendor_Item_Number`   | (extended)            | Vendor part number             |
| `Department`           | `category`            | 2-digit department code        |
| `Department_Name`      | `category` (name)     | Department description         |
| `Class`                | `category` (class)    | Class within department        |
| `Fineline`             | `category` (fine)     | Fineline sub-class             |
| `Qty_On_Hand`          | `quantity`            | Current stock                  |
| `Unit_Cost`            | `cost`                | Net member cost                |
| `Retail_Price`         | `revenue`             | Current retail                 |
| `Suggested_Retail`     | `revenue` (msrp)      | Do It Best suggested retail    |
| `Margin_Pct`           | `margin`              | Gross margin %                 |
| `Qty_On_Order`         | (extended)            | On order qty                   |
| `Min_Stock`            | `reorder_point`       | Minimum stock level            |
| `Max_Stock`            | `reorder_point` (max) | Maximum stock level            |
| `Reorder_Qty`          | (extended)            | Suggested reorder qty          |
| `Last_Receipt_Date`    | `last_purchase_date`  | MM/DD/YYYY                     |
| `Last_Sale_Date`       | `last_sale_date`      | MM/DD/YYYY                     |
| `YTD_Sales_Units`      | `sold`                | Year-to-date units             |
| `YTD_Sales_Dollars`    | (extended)            | YTD dollar sales               |
| `YTD_Cost`             | (extended)            | YTD COGS                       |
| `Whse_Code`            | `location` (wh)       | Warehouse: IN, OH, MT, OR, PA, CT |
| `Pack_Size`            | `package_qty`         | Case pack qty                  |
| `Velocity_Code`        | (extended)            | Sales velocity tier            |
| `Catalog_Page`         | (extended)            | Do It Best catalog page #      |

#### Patronage Report Fields

| Column                         | Description                     |
|--------------------------------|---------------------------------|
| `Patronage_Rate_Regular`       | 11.11% (FY2024)                |
| `Patronage_Rate_Promotional`   | 5.84%                          |
| `Patronage_Rate_DirectShip`    | 1.06%                          |
| `Patronage_Rate_Lumber`        | 0.74%                          |
| `Cash_Discount_Earned`         | 2% on warehouse invoices       |
| `Total_Rebate_Earned`          | Combined patronage + discounts |

---

### 4.2 Ace Hardware (ARS)

**Vendor:** Ace Hardware Corporation (Oak Brook, IL)
**System:** Ace Retail Systems (ARS) / ACENET / Eagle Vision
**Confidence:** MEDIUM (member portal, EDI 852 for inventory)

| Ace Column             | → Canonical Field     | Notes                          |
|------------------------|-----------------------|--------------------------------|
| `Store_Number`         | `location`            | 5-digit Ace store number       |
| `SKU`                  | `sku`                 | Ace internal SKU               |
| `Ace_Item_Number`      | `sku` (ace)           | 7-digit Ace catalog number     |
| `UPC`                  | `sku` (upc)           | UPC barcode                    |
| `Description`          | `description`         | Item description               |
| `Dept`                 | `category`            | 2-digit department (01-99)     |
| `Dept_Description`     | `category` (name)     | Department name                |
| `Class`                | `category` (class)    | Class within department        |
| `Subclass`             | `category` (sub)      | Subclass code                  |
| `Vendor_Number`        | `vendor`              | Vendor / supplier number       |
| `Vendor_Name`          | `vendor` (name)       | Vendor name                    |
| `Vendor_Part_Number`   | (extended)            | Manufacturer part number       |
| `On_Hand`              | `quantity`            | Current QOH                    |
| `On_Order`             | (extended)            | On order qty                   |
| `Unit_Cost`            | `cost`                | Cost per unit                  |
| `Retail`               | `revenue`             | Retail price                   |
| `Margin`               | `margin`              | Margin percentage              |
| `Last_Sold_Date`       | `last_sale_date`      | MM/DD/YYYY                     |
| `Last_Received_Date`   | `last_purchase_date`  | MM/DD/YYYY                     |
| `YTD_Sold_Qty`         | `sold`                | YTD units sold                 |
| `YTD_Sold_Dollars`     | (extended)            | YTD dollar sales               |
| `Min`                  | `reorder_point`       | Minimum stock                  |
| `Max`                  | `reorder_point` (max) | Maximum stock                  |
| `Bin_Location`         | `location` (bin)      | Shelf / bin location           |
| `Pack_Size`            | `package_qty`         | Case pack qty                  |
| `Season_Code`          | (extended)            | Seasonal classification        |
| `Status`               | `status`              | ACTIVE/DISCONTINUED/SEASONAL/CLOSEOUT |

---

### 4.3 True Value

**Vendor:** True Value Company (now owned by Do It Best as of 2024)
**Confidence:** MEDIUM (legacy format, transitioning to Do It Best)

| True Value Column     | → Canonical Field     | Notes                          |
|-----------------------|-----------------------|--------------------------------|
| `Store_Number`        | `location`            | 6-digit True Value store #     |
| `Item_Number`         | `sku`                 | True Value catalog item #      |
| `TV_SKU`              | `sku` (tv)            | ⚠ True Value internal SKU     |
| `UPC`                 | `sku` (upc)           | UPC barcode                    |
| `Description`         | `description`         | Product description            |
| `Vendor_Number`       | `vendor`              | Vendor ID                      |
| `Vendor_Name`         | `vendor` (name)       | Vendor name                    |
| `Vendor_Part_Number`  | (extended)            | Manufacturer part number       |
| `Department`          | `category`            | Department code                |
| `Dept_Name`           | `category` (name)     | Department name                |
| `Category`            | `category` (sub)      | Category code                  |
| `Qty_On_Hand`         | `quantity`            | Current stock                  |
| `Unit_Cost`           | `cost`                | Unit cost                      |
| `Retail_Price`        | `revenue`             | Retail price                   |
| `Sug_Retail`          | `revenue` (msrp)      | TV suggested retail            |
| `Margin_Pct`          | `margin`              | Margin percentage              |
| `Qty_On_Order`        | (extended)            | On order qty                   |
| `Min_Level`           | `reorder_point`       | Minimum stock level            |
| `Max_Level`           | `reorder_point` (max) | Maximum stock level            |
| `Last_Sale`           | `last_sale_date`      | Last sale date                 |
| `Last_Receipt`        | `last_purchase_date`  | Last receipt date              |
| `Sales_YTD_Units`     | `sold`                | YTD units sold                 |
| `Sales_YTD_Dollars`   | (extended)            | YTD dollar sales               |
| `Pack_Qty`            | `package_qty`         | Pack / case qty                |
| `Warehouse`           | `location` (wh)       | True Value warehouse code      |
| `Status`              | `status`              | Active/Discontinued/Clearance  |

> ⚠ **Note:** True Value was acquired by Do It Best in 2024. Legacy TV formats
> will eventually migrate to Do It Best format. Support both during transition.

---

### 4.4 Orgill

**Vendor:** Orgill Inc. (Memphis, TN)
**Confidence:** VERY HIGH (PO parser fully implemented)

#### PO Line Item Columns (28 fields per row)

| Position | Column Header             | → Canonical Field     |
|----------|---------------------------|-----------------------|
| 0        | `Line`                    | (sequence)            |
| 1        | `Retail`                  | `revenue`             |
| 2        | `Item`                    | `sku`                 |
| 3        | (flag Y/blank)            | (extended)            |
| 4        | `Ord Qty`                 | `quantity` (ordered)  |
| 5        | `Unit`                    | `package_qty` (uom)   |
| 6        | `Description`             | `description`         |
| 7        | `Unit Cost`               | `cost`                |
| 8        | `Spc`                     | (extended: `*` = special price) |
| 9        | `Ext Cost`                | `sub_total`           |
| 10       | `Prod Care`               | (extended)            |
| 11       | `Qty Fill`                | `quantity` (shipped)  |
| 12       | `Out`                     | (extended: `*` = short ship) |
| 13       | `Shelf Pk`                | `package_qty`         |
| 14       | `UPC Code`                | `sku` (upc)           |
| 15       | `POS`                     | (extended)            |
| 16       | `Crd Inv`                 | `transaction_id` (credit) |
| 17       | `Retail Dept Description` | `category`            |
| 18       | `Dept`                    | `category` (code)     |
| 19       | `Vendor Item Num`         | (extended)            |
| 20       | `PickLne`                 | (extended)            |
| 21       | `Country of Origin`       | (extended)            |
| 22       | `Resvd Qty`               | (extended)            |
| 23       | `Xref Item`               | `sku` (xref)          |
| 24       | `Itm Weight`              | (extended)            |
| 25       | `French Description`      | `description` (fr)    |
| 26       | `Cust Item Desc`          | `description` (custom)|
| 27       | `Velocity Code`           | (extended)            |
| 28       | `Rtl Sens Code`           | (extended)            |

**Velocity Codes:** A=High, B=Good, C=Medium, D=Slow, F=Very slow, N=New item

#### Orgill PO Header (rows 1-14)

```
Row 1:  "Shipto: {id} ,Billto: {id} ,Order: {po_number}"
Row 3:  "Date: MM/DD/YYYY"
Row 4:  "Status: INVOICED|PENDING|OPEN"
Row 7:  "USD Amount: {amount}"
Row 15: Column headers for line items
```

---

## 5. EDI Standards (ANSI X12)

### EDI 846 — Inventory Inquiry/Advice

The primary EDI transaction for inventory data exchange in hardware retail.

| Segment | Qualifier | → Canonical Field     | Description                |
|---------|-----------|----------------------|----------------------------|
| `LIN`   | `UP`      | `sku` (upc)          | UPC barcode                |
| `LIN`   | `VP`      | `vendor` (part)       | Vendor part number         |
| `LIN`   | `SK`      | `sku`                 | SKU                        |
| `LIN`   | `IN`      | `sku` (buyer)         | Buyer item number          |
| `PID`   | `F`       | `description`         | Free-form description      |
| `QTY`   | `33`      | `quantity`            | Available for sale         |
| `QTY`   | `QH`      | `quantity` (on hand)  | Quantity on hand           |
| `QTY`   | `QO`      | (extended)            | Quantity on order          |
| `QTY`   | `59`      | `reorder_point`       | Minimum quantity           |
| `QTY`   | `60`      | `reorder_point` (max) | Maximum quantity           |
| `QTY`   | `17`      | `sold`                | Quantity sold              |
| `AMT`   | `UC`      | `cost`                | Unit cost                  |
| `AMT`   | `RT`      | `revenue`             | Retail price               |
| `AMT`   | `SU`      | `revenue` (msrp)      | Suggested retail           |
| `AMT`   | `IV`      | `sub_total`           | Inventory value            |
| `DTM`   | `050`     | `last_purchase_date`  | Last receipt date          |
| `DTM`   | `511`     | `last_sale_date`      | Last sale date             |
| `REF`   | `DP`      | `category`            | Department number          |
| `REF`   | `VN`      | `vendor`              | Vendor number              |
| `REF`   | `WH`      | `location`            | Warehouse                  |

### EDI 850 — Purchase Order

| Segment | Field  | → Canonical Field     | Description              |
|---------|--------|-----------------------|--------------------------|
| `BEG`   | `03`   | `transaction_id`      | PO number                |
| `BEG`   | `05`   | `date`                | PO date (CCYYMMDD)       |
| `PO1`   | `02`   | `quantity` (ordered)  | Quantity ordered          |
| `PO1`   | `04`   | `cost`                | Unit price               |
| `PO1`   | `07`   | `sku`                 | Product ID               |

### EDI 810 — Invoice

| Segment | Field | → Canonical Field     | Description               |
|---------|-------|-----------------------|---------------------------|
| `BIG`   | `01`  | `date`                | Invoice date              |
| `BIG`   | `02`  | `transaction_id`      | Invoice number            |
| `IT1`   | `02`  | `quantity` (invoiced) | Quantity invoiced         |
| `IT1`   | `04`  | `cost`                | Unit price                |
| `IT1`   | `07`  | `sku`                 | Product ID                |
| `TDS`   | `01`  | `sub_total`           | Total amount (in cents)   |

### EDI 856 — Advance Ship Notice

| Segment | Field | → Canonical Field     | Description               |
|---------|-------|-----------------------|---------------------------|
| `BSN`   | `02`  | `transaction_id`      | Shipment ID / BOL         |
| `BSN`   | `03`  | `date`                | Ship date                 |
| `SN1`   | `02`  | `quantity` (shipped)  | Units shipped             |
| `SN1`   | `05`  | `quantity` (ordered)  | Units ordered (for variance) |

---

## 6. Auto-Detection Fingerprints

Each POS system has unique column names that serve as fingerprints for automatic
detection. The adapter should check for these signatures to identify the source
system without user input.

### Detection Priority Order

```python
DETECTION_RULES = [
    # Tier 1 — Unique column names (1 column = definitive match)
    {
        "system": "paladin_native",
        "unique": {"partnumber", "mktcost", "plvl_price1", "stockonhand"},
        "confidence": "definitive",  # ANY one of these = Paladin native
    },
    {
        "system": "paladin_report",
        "unique": {"Margin @Cost", "Inventory @Cost", "Inventory @Retail",
                   "Inventory @AvgCost", "Retail Dif."},
        "confidence": "definitive",
    },
    {
        "system": "epicor_eagle",
        "unique": {"qty_oh", "posting_cost"},  # Both present = Eagle
        "min_match": 2,
        "confidence": "high",
    },
    {
        "system": "ncr_counterpoint",
        "unique": {"item_no", "categ_cod", "subcat_cod", "qty_on_hnd",
                   "lst_cost", "prc_1"},
        "min_match": 3,
        "confidence": "high",
    },
    {
        "system": "retail_pro",
        "unique": {"SID", "ALU", "DCS"},
        "min_match": 2,
        "confidence": "high",
    },
    {
        "system": "korona",
        "unique": {"Commodity Group", "Organizational Unit", "Optimal Stock",
                   "Sector", "Assortment"},
        "min_match": 2,
        "confidence": "high",
    },
    {
        "system": "lightspeed_r",
        "unique": {"system_id", "custom_sku", "default_cost",
                   "variant_inventory_qty", "manufacturer_sku"},
        "min_match": 3,
        "confidence": "high",
    },
    {
        "system": "lightspeed_x",
        "unique": {"product_handle", "supply_price", "outlet_name",
                   "variant_loyalty_value"},
        "min_match": 2,
        "confidence": "high",
    },

    # Tier 2 — Combination-based detection (2+ columns required)
    {
        "system": "idosoft",
        "unique": {"Qty.", "Margin @Cost", "Inventory @Cost"},
        "strong": {"wholesale", "qty"},
        "min_match": 2,
        "confidence": "medium",
    },
    {
        "system": "microbiz",
        "unique": {"Item Number", "Item Name", "Track Inventory",
                   "Compare At Price", "Quantity Reserved"},
        "min_match": 3,
        "confidence": "medium",
    },
    {
        "system": "rain_pos",
        "unique": {"Product ID", "Catalog ID", "Serialized", "Online Price"},
        "min_match": 2,
        "confidence": "medium",
    },
    {
        "system": "shopify",
        "unique": {"handle", "variant_sku", "variant_barcode",
                   "variant_price", "body_html"},
        "min_match": 2,
        "confidence": "high",
    },
    {
        "system": "square",
        "unique": {"token", "reference_handle", "gtin", "new_quantity"},
        "min_match": 2,
        "confidence": "high",
    },
    {
        "system": "dynamics_rms",
        "unique": {"itemlookupcode", "pricea", "priceb", "pricec"},
        "min_match": 2,
        "confidence": "high",
    },

    # Tier 3 — Co-op systems (require context + column names)
    {
        "system": "do_it_best",
        "unique": {"Member_Number", "Whse_Code", "Velocity_Code",
                   "Catalog_Page"},
        "min_match": 2,
        "confidence": "medium",
    },
    {
        "system": "ace_hardware",
        "unique": {"Ace_Item_Number", "Store_Number", "Season_Code"},
        "min_match": 2,
        "confidence": "medium",
    },
    {
        "system": "true_value",
        "unique": {"TV_SKU", "Store_Number", "Sug_Retail"},
        "min_match": 2,
        "confidence": "medium",
    },
    {
        "system": "orgill",
        "unique": {"Ord Qty", "Shelf Pk", "Prod Care", "Velocity Code",
                   "PickLne", "Rtl Sens Code"},
        "min_match": 3,
        "confidence": "high",
    },
]
```

---

## 7. Format Specifications

### Date Formats by System

| POS System        | Primary Format    | Secondary Format   | Notes              |
|-------------------|-------------------|--------------------|--------------------|
| Paladin (native)  | `YYYYMMDD`        | `Mon DD,YY`        | SHLP uses Mon DD,YY |
| Epicor Eagle      | `MM/DD/YY`        | `YYYY-MM-DD`       | Import Designer    |
| NCR Counterpoint  | `YYYY-MM-DD`      | `MM/DD/YYYY`       | SQL Server default |
| Lightspeed        | ISO 8601          | `MM/DD/YYYY`       | API vs CSV differ  |
| Retail Pro        | `MM/DD/YYYY`      |                    | US default         |
| KORONA            | ISO 8601          | `MM/DD/YYYY`       | API vs CSV differ  |
| Do It Best        | `MM/DD/YYYY`      |                    |                    |
| Ace Hardware      | `MM/DD/YYYY`      |                    |                    |
| True Value        | `MM/DD/YYYY`      |                    |                    |
| Orgill            | `MM/DD/YYYY`      |                    | Header only        |
| EDI (all)         | `CCYYMMDD`        |                    | 8-digit, no sep    |

### Delimiter Conventions

| POS System        | Default Delimiter | Notes                           |
|-------------------|-------------------|---------------------------------|
| Paladin           | Comma (CSV)       | Also supports Tab               |
| Epicor Eagle      | Pipe (`\|`)       | DAFL default; CSV via reports   |
| NCR Counterpoint  | Tab               | Also CSV via Crystal Reports    |
| Lightspeed        | Comma (CSV)       |                                 |
| Retail Pro        | Comma (CSV)       |                                 |
| KORONA            | Comma (CSV)       |                                 |
| Do It Best        | Comma (CSV)       |                                 |
| Ace Hardware      | Comma (CSV)       |                                 |
| Orgill            | Comma (CSV)       | POs have fixed-width header     |
| EDI               | `*` (element)     | `~` (segment), `>` (component) |

---

## 8. Known Quirks & Caveats

### Critical Issues

1. **Paladin margin is unreliable.** Always recalculate margin from cost and price.
   Reference: `apps/api/src/services/analysis.py` lines 1029, 1045.

2. **Paladin "Qty." vs "Qty On Hand" distinction.** `Qty.` includes negatives
   (items sold but not received); `Qty On Hand` is physical count, never negative.
   For negative inventory detection, `Qty.` is the critical field.
   Reference: sample store adapter lines 226-233.

3. **IdoSoft large file sizes.** Sample store custom_1.csv has 156K+ rows. Use
   streaming/chunked reading, not `pd.read_csv()` with full load.

4. **Epicor Eagle commas in data.** Eagle's DAFL import converts commas to `?`.
   If you see `?` in descriptions, the source is likely Eagle.

5. **NCR Counterpoint abbreviated columns.** All columns use SQL-style
   abbreviations (`qty_on_hnd` not `quantity_on_hand`, `lst_cost` not `last_cost`).
   Matching must handle these abbreviation patterns.

6. **BisTrack multi-UOM complexity.** A single item can have 4 different units
   of measure (selling, buying, pricing, stocking). Conversion factors are required
   for accurate quantity comparisons.

7. **Do It Best 7-digit item numbers.** These are Do It Best catalog numbers,
   NOT UPCs. The UPC is a separate field. Don't confuse them.

8. **Ace Hardware 7-digit Ace Item Numbers.** Same caveat as Do It Best — these
   are Ace catalog IDs, not UPCs.

9. **True Value data transition.** Do It Best acquired True Value in 2024. Legacy
   TV_SKU fields will eventually transition to Do It Best Item_Number format.
   Support both formats during migration period.

10. **Orgill PO header is 14 rows.** Line items start at row 15+. The parser must
    skip the structured header block. Reference: `adapters/orgill/po_parser.py`.

### Encoding Issues

| System          | Common Encoding | Watch For                          |
|-----------------|----------------|------------------------------------|
| Paladin         | UTF-8, CP1252   | Special chars in descriptions      |
| Epicor Eagle    | UTF-8, CP1252   | `?` replacing commas in DAFL       |
| NCR Counterpoint| UTF-8           | Generally clean                    |
| IdoSoft         | UTF-8           | Use `errors="replace"` for safety  |
| Orgill          | UTF-8           | French descriptions in column 25   |

### Column Name Normalization Rules

For reliable matching, normalize column names before lookup:

```python
def normalize_column(name: str) -> str:
    """Normalize a column name for matching."""
    name = name.strip().lower()
    name = name.replace(" ", "_").replace(".", "").replace("#", "")
    name = name.replace("@", "at_")
    # Remove trailing underscores
    name = name.rstrip("_")
    return name
```

---

## Appendix: Files Where Mappings Are Defined

| File | Purpose | Aliases |
|------|---------|---------|
| `config/pos_mappings/standard_fields.yaml` | **Primary source of truth** | 400+ |
| `config/pos_mappings/field_importance.yaml` | Field ratings (1-5) | 27 fields |
| `config/pos_mappings/supported_systems.yaml` | Supported POS list | 15+ |
| `apps/api/src/services/mapping.py` | AI + heuristic matching service | (uses YAML) |
| `apps/api/src/utils/column_mappings.py` | YAML config loader | (loads YAML) |
| `packages/sentinel-engine/src/sentinel_engine/core.py` | VSA alias lists | ~180 |
| `apps/api/src/routes/analysis.py` | NUMERIC_COLUMN_ALIASES flat list | ~150 |
| `scripts/hardware_inventory_analyzer.py` | COLUMN_ALIASES dict | ~100 |
| `profit-sentinel-rs/python/sentinel_agent/adapters/sample_store/inventory.py` | Sample store/IdoSoft adapter | 35+ |
| `profit-sentinel-rs/python/sentinel_agent/adapters/orgill/po_parser.py` | Orgill PO parser | 28 |

> ⚠ **Consolidation needed:** Column aliases exist in 3+ locations (YAML, core.py,
> analysis.py, hardware_inventory_analyzer.py). The YAML config should be the single
> source of truth, with all other locations importing from it.
