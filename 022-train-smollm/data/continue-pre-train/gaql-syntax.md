# Google Ads Query Language (GAQL) Syntax Reference

## Overview

The Google Ads Query Language (GAQL) is used to query the Google Ads API to retrieve:

1. **Resources** and their related attributes, segments, and metrics using `GoogleAdsService.Search` or `GoogleAdsService.SearchStream`
2. **Metadata** about available fields and resources using `GoogleAdsFieldService`

## GAQL Grammar Reference

Formal grammar in regular expression notation:

```
Query            -> SelectClause FromClause WhereClause? OrderByClause?
                    LimitClause? ParametersClause?
SelectClause     -> SELECT FieldName (, FieldName)*
FromClause       -> FROM ResourceName
WhereClause      -> WHERE Condition (AND Condition)*
OrderByClause    -> ORDER BY Ordering (, Ordering)*
LimitClause      -> LIMIT PositiveInteger
ParametersClause -> PARAMETERS Literal = Value (, Literal = Value)*

Condition        -> FieldName Operator Value
Operator         -> = | != | > | >= | < | <= | IN | NOT IN |
                    LIKE | NOT LIKE | CONTAINS ANY | CONTAINS ALL |
                    CONTAINS NONE | IS NULL | IS NOT NULL | DURING |
                    BETWEEN | REGEXP_MATCH | NOT REGEXP_MATCH
Value            -> Literal | LiteralList | Number | NumberList | String |
                    StringList | Function
Ordering         -> FieldName (ASC | DESC)?

FieldName        -> [a-z] ([a-zA-Z0-9._])*
ResourceName     -> [a-z] ([a-zA-Z_])*

StringList       -> ( String (, String)* )
LiteralList      -> ( Literal (, Literal)* )
NumberList       -> ( Number (, Number)* )

PositiveInteger  -> [1-9] ([0-9])*
Number           -> -? [0-9]+ (. [0-9] [0-9]*)?
String           -> (' Char* ') | (" Char* ")
Literal          -> [a-zA-Z0-9_]*

Function         -> LAST_14_DAYS | LAST_30_DAYS | LAST_7_DAYS |
                    LAST_BUSINESS_WEEK | LAST_MONTH | LAST_WEEK_MON_SUN |
                    LAST_WEEK_SUN_SAT | THIS_MONTH | THIS_WEEK_MON_TODAY |
                    THIS_WEEK_SUN_TODAY | TODAY | YESTERDAY
```

`?` = optional, `*` = zero or more, `+` = one or more, `(…)` = grouping, `[a-z0-9]` = character ranges, `|` = "or".

### Grammar rules and limitations

- `REGEXP_MATCH` uses [RE2 syntax](https://github.com/google/re2/wiki/Syntax).
- To match a literal `[`, `]`, `%`, or `_` with `LIKE`, surround the character in square brackets: `campaign.name LIKE '[[]Earth[_]to[_]Mars[]]%'`
- `LIKE` can only be used on a string field, not an array.

---

## Status Filtering

**Do NOT add entity status filters** (`campaign.status`, `ad_group.status`, `ad_group_ad.status`, etc.) to analysis or reporting queries. Campaigns that are currently REMOVED or PAUSED may have had active spend during the queried date range — filtering them out silently drops real historical data.

Only filter by status when the user **explicitly asks** (e.g., "show only enabled campaigns", "exclude removed campaigns").

Status filtering IS appropriate for **operational/management** queries (listing currently active campaigns for budget changes, pausing, etc.) — not for performance analysis.

---

## Pre-aggregate in the query (quota-conscious)

**Return aggregated reports, not raw data.** The API has query quotas. To minimize queries and row count:

- **Omit `segments.date`** unless the user explicitly asks for daily/weekly/monthly trends. Without it, metrics are summed over the WHERE date range — one row per entity (campaign, product, etc.).
- **Include `segments.date`** only when you need a time series (e.g. daily spend trend). Each date multiplies rows.
- Design the SELECT to match the report level you need: campaign totals, product totals, ad group totals — not raw daily rows.
- Prefer one query with the right aggregation over multiple queries or post-processing.

## Google Ads API Limitations

Be aware of quotas and limits. Exceeding them returns `RESOURCE_EXHAUSTED`, `RESOURCE_TEMPORARILY_EXHAUSTED`, or `FILTER_HAS_TOO_MANY_VALUES`.

### Daily operation quota (per developer token)

| Access level | Production accounts | Test accounts |
|--------------|---------------------|---------------|
| Explorer     | 2,880 operations/day | 15,000/day    |
| Basic        | 15,000/day          | 15,000/day    |
| Standard     | Unlimited           | Unlimited     |

- **One Search or SearchStream request = 1 operation** regardless of streaming, pagination, or row count.
- Paginated requests with a valid `next_page_token` do **not** count against the quota; invalid/expired tokens do.
- Failed requests that return `GoogleAdsFailure` **still count** against the quota.

### Response size

- gRPC message limit: **64 MB**. Large SELECTs or many rows can exceed it.
- Mitigation: select fewer fields, use `SearchStream`, or add `LIMIT` to reduce rows.
- Exceeding → `429 Resource Exhausted`.

### GAQL query limits

- **IN clause**: Maximum **20,000 values**. Exceeding → `FILTER_HAS_TOO_MANY_VALUES`. Split into multiple queries if needed.
- **Complex queries**: Very large or complex queries may trigger `INTERNAL_ERROR` even within stated limits. Simplify or split the query.

### Rate limits (QPS)

- Token-bucket rate limiting per customer ID and developer token.
- Exceeding → `RESOURCE_TEMPORARILY_EXHAUSTED`. Throttle or queue requests.

### Best practices

- Pre-aggregate in the query (omit segments unless trend data is needed).
- Use `LIMIT` (e.g. 25–100) for exploratory or UI-bound reports. **For full-data queries** (all products, complete categorization, entire catalog, etc.) **omit LIMIT** — the backend paginates automatically and returns all rows.
- Select only required fields.
- Prefer one well-designed query over many small ones.
- **Prefer native computed metrics** — SELECT `metrics.cost_per_conversion` (CPA), `metrics.ctr`, `metrics.average_cpc` — these work on **all** resources. For ROAS (`metrics.conversions_value_per_cost`), it is **only supported on `keyword_view` and `search_term_view`** — prohibited on campaign, ad_group, ad_group_ad, shopping_performance_view, product_group_view. AOV (`metrics.average_order_value_micros`) is prohibited on search_term_view. Impression share metrics have per-resource restrictions. See full matrix in `google-api-routing.md`.

## Field Categories

Understanding field categories is essential for building effective GAQL queries:

1. **RESOURCE**: Represents a primary entity (e.g., `campaign`, `ad_group`) that can be used in the FROM clause
2. **ATTRIBUTE**: Properties of a resource (e.g., `campaign.id`, `campaign.name`). Including these may segment results depending on the resource relationship
3. **SEGMENT**: Fields that always segment search queries (e.g., `segments.date`, `segments.device`)
4. **METRIC**: Performance data fields (e.g., `metrics.impressions`, `metrics.clicks`) that never segment search queries

## Query Structure

A GAQL query consists of the following components:

```
SELECT
  <field_1>,
  <field_2>,
  ...
FROM <resource>
WHERE <condition_1> AND <condition_2> AND ...
ORDER BY <field_1> [ASC|DESC], <field_2> [ASC|DESC], ...
LIMIT <number_of_results>
```

### SELECT Clause

The `SELECT` clause specifies the fields to return in the query results:

```
SELECT
  campaign.id,
  campaign.name,
  metrics.impressions,
  segments.device
```

Only fields that are marked as `selectable: true` in the `GoogleAdsField` metadata can be used in the SELECT clause.

### FROM Clause

The `FROM` clause specifies the primary resource type to query from. Only one resource can be specified, and it must have the category `RESOURCE`.

```
FROM campaign
```

### WHERE Clause (optional)

The `WHERE` clause specifies conditions to filter the results. Only fields marked as `filterable: true` in the `GoogleAdsField` metadata can be used for filtering.

```
WHERE 
  campaign.status = 'ENABLED'
  AND metrics.impressions > 1000
  AND segments.date DURING LAST_30_DAYS
```

**WHERE only supports flat AND conditions.** Every condition is joined with `AND` — no `OR`, no parentheses, no grouping. When you need OR-style logic, run separate queries and merge results client-side, or use a single broader filter and narrow down post-query.

### ORDER BY Clause (optional)

The `ORDER BY` clause specifies how to sort the results. Only fields marked as `sortable: true` in the `GoogleAdsField` metadata can be used for sorting.

```
ORDER BY metrics.impressions DESC, campaign.id
```

### LIMIT Clause (optional)

The `LIMIT` clause restricts the number of results returned.

```
LIMIT 100
```

## Field Metadata Exploration

To explore available fields and their properties, use the `GoogleAdsFieldService`:

```
SELECT
  name,
  category,
  selectable,
  filterable,
  sortable,
  selectable_with,
  attribute_resources,
  metrics,
  segments,
  data_type,
  enum_values,
  is_repeated
WHERE name = "campaign.id"
```

Key metadata properties to understand:

- **`selectable`**: Whether the field can be used in a SELECT clause
- **`filterable`**: Whether the field can be used in a WHERE clause
- **`sortable`**: Whether the field can be used in an ORDER BY clause
- **`selectable_with`**: Lists resources, segments, and metrics that are selectable with this field
- **`attribute_resources`**: For RESOURCE fields, lists the resources that are selectable with this resource and don't segment metrics
- **`metrics`**: For RESOURCE fields, lists metrics that are selectable when this resource is in the FROM clause
- **`segments`**: For RESOURCE fields, lists fields that segment metrics when this resource is used in the FROM clause
- **`data_type`**: Determines which operators can be used with the field in WHERE clauses
- **`enum_values`**: Lists possible values for ENUM type fields
- **`is_repeated`**: Whether the field can contain multiple values

## Data Types and Operators

Different field data types support different operators in WHERE clauses:

### String Fields
- `=`, `!=`, `IN`, `NOT IN`
- `LIKE`, `NOT LIKE` (case-sensitive string matching)
- `CONTAINS ANY`, `CONTAINS ALL`, `CONTAINS NONE` (for repeated fields)

### Numeric Fields
- `=`, `!=`, `<`, `<=`, `>`, `>=`
- `IN`, `NOT IN`

### Date Fields
- `=`, `!=`, `<`, `<=`, `>`, `>=`
- `DURING` (with named date ranges)
- `BETWEEN` (with date literals)

### Enum Fields
- `=`, `!=`, `IN`, `NOT IN`
- Values must match exactly as listed in `enum_values`

### Boolean Fields
- `=`, `!=`
- Values must be `TRUE` or `FALSE`

## Date Ranges

### Literal Date Ranges (custom date range)
Use `BETWEEN` with ISO dates. For a specific month or date range:
```
WHERE segments.date BETWEEN '2026-01-01' AND '2026-01-31'
```
Alternative with comparison operators:
```
WHERE segments.date >= '2026-01-01' AND segments.date <= '2026-01-31'
```

### Named Date Ranges

Valid DURING literals (**only these are accepted by the Google Ads API**):
```
WHERE segments.date DURING LAST_7_DAYS
WHERE segments.date DURING LAST_14_DAYS
WHERE segments.date DURING LAST_30_DAYS
WHERE segments.date DURING THIS_MONTH
WHERE segments.date DURING LAST_MONTH
WHERE segments.date DURING THIS_QUARTER
WHERE segments.date DURING LAST_QUARTER
```

Other valid but less common: `TODAY`, `YESTERDAY`, `THIS_WEEK_SUN_TODAY`, `THIS_WEEK_MON_TODAY`, `LAST_WEEK_SUN_SAT`, `LAST_WEEK_MON_SUN`, `LAST_BUSINESS_WEEK`.

**LAST_60_DAYS, LAST_90_DAYS, etc. do NOT exist** — using them causes `INVALID_VALUE_WITH_DURING_OPERATOR`. For ranges beyond 30 days, use `BETWEEN`:
```
WHERE segments.date BETWEEN '2026-01-01' AND '2026-03-01'
```

**Critical:** `DURING` accepts only the named literals listed above. `DURING ['date1','date2']` is also invalid — use `BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'` for custom ranges.

### Predefined Time Period Segments

Some date segments refer to a predefined period of time:

- **`segments.week`** — returns the Monday (ISO week start) of each week
- **`segments.month`** — returns the first day of each month
- **`segments.quarter`** — returns the first day of each quarter

When included in **SELECT**, metrics are automatically segmented by that period — one row per entity per period. This eliminates the need for multiple queries when analyzing across multiple weeks/months.

When **filtering** on these segments, use `=` with the first day of the period. Specifying a date that is not the first day causes a `MISALIGNED_DATE_FOR_FILTER` error:
```
segments.month = '2026-05-01'
segments.week = '2026-02-03'
```

**Multi-week analysis pattern:** Include `segments.week` in SELECT with a single `BETWEEN` date range in WHERE to get per-entity-per-week data in one query:
```
SELECT campaign.name, segments.week, metrics.cost_micros, metrics.clicks, metrics.conversions
FROM campaign
WHERE segments.date BETWEEN '2026-02-01' AND '2026-03-07'
```
This returns one row per campaign per week — no need for separate queries per week.

### Date Functions
```
WHERE segments.date = YESTERDAY
WHERE segments.date = TODAY
```

## Case Sensitivity Rules

1. **Field and resource names**: Case-sensitive (`campaign.id` not `Campaign.Id`)
2. **Enumeration values**: Case-sensitive (`'ENABLED'` not `'enabled'`)
3. **String literals in conditions**:
   - Case-insensitive by default (`WHERE campaign.name = 'brand campaign'`)
   - Use `LIKE` for case-sensitive matching (`WHERE campaign.name LIKE 'Brand Campaign'`)

## Ordering and Limiting Results

### Ordering
- Results can be ordered by one or more fields
- Use `ASC` (default) or `DESC` to specify direction
- Only fields marked as `sortable: true` can be used

```
ORDER BY metrics.impressions DESC, campaign.id ASC
```

### Limiting
- Use LIMIT to restrict the number of rows returned
- Always use ORDER BY with LIMIT for consistent pagination
- The maximum value is system-dependent

```
LIMIT 100
```

## Query Examples

### Basic Campaign Query
```
SELECT
  campaign.id,
  campaign.name,
  campaign.status
FROM campaign
ORDER BY campaign.id
```

### Query with Metrics and Filtering
```
SELECT
  campaign.id,
  campaign.name,
  metrics.impressions,
  metrics.clicks,
  metrics.cost_micros
FROM campaign
WHERE 
  campaign.status = 'ENABLED'
  AND metrics.impressions > 1000
ORDER BY metrics.impressions DESC
LIMIT 10
```

### Aggregated report (no segments — preferred when totals suffice)
```
SELECT campaign.id, campaign.name, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions
FROM campaign
WHERE segments.date DURING LAST_30_DAYS
ORDER BY metrics.cost_micros DESC
LIMIT 25
```
Date filter in WHERE aggregates over the range; no segments.date in SELECT = one row per campaign.

### Query with Segments (use only when daily/trend breakdown is needed)
```
SELECT
  campaign.id,
  campaign.name,
  segments.date,
  metrics.impressions,
  metrics.clicks,
  metrics.conversions
FROM campaign
WHERE 
  segments.date DURING LAST_30_DAYS
  AND campaign.status = 'ENABLED'
ORDER BY segments.date DESC, metrics.impressions DESC
```

### Query with Attributed Resources
```
SELECT
  campaign.id,
  campaign.name,
  campaign.status,
  bidding_strategy.id,
  bidding_strategy.name,
  bidding_strategy.type
FROM campaign
WHERE campaign.status = 'ENABLED'
```

### Field Metadata Query
```
SELECT
  name,
  category,
  selectable,
  filterable,
  sortable,
  data_type,
  enum_values
WHERE name LIKE 'campaign.%'
```

## Asset Queries

### Asset Entity Queries

Query the `asset` entity for asset attributes:

```
SELECT
  asset.id,
  asset.name,
  asset.resource_name,
  asset.type
FROM asset
```

### Type-Specific Asset Attributes

Assets have type-specific attributes that can be queried based on their type:

```
SELECT
  asset.id,
  asset.name,
  asset.resource_name,
  asset.youtube_video_asset.youtube_video_id
FROM asset
WHERE asset.type = 'YOUTUBE_VIDEO'
```

### Asset Metrics at Different Levels

Asset metrics are available through three main resources:

1. **ad_group_asset**: Asset metrics at the ad group level
2. **campaign_asset**: Asset metrics at the campaign level
3. **customer_asset**: Asset metrics at the customer level

Example of querying ad-group level asset metrics:

```
SELECT
  ad_group.id,
  asset.id,
  metrics.clicks,
  metrics.impressions
FROM ad_group_asset
WHERE segments.date DURING LAST_MONTH
ORDER BY metrics.impressions DESC
```

### Ad-Level Asset Performance

Ad-level performance metrics for assets are aggregated in the `ad_group_ad_asset_view`.

**Note**: The `ad_group_ad_asset_view` only returns information for assets related to App ads.

This view includes the `performance_label` attribute with the following possible values:
- `BEST`: Best performing assets
- `GOOD`: Good performing assets
- `LOW`: Worst performing assets
- `LEARNING`: Asset has impressions but stats aren't statistically significant yet
- `PENDING`: Asset doesn't have performance information yet (may be under review)
- `UNKNOWN`: Value unknown in this version
- `UNSPECIFIED`: Not specified

Example query for ad-level asset performance:

```
SELECT
  ad_group_ad_asset_view.ad_group_ad,
  ad_group_ad_asset_view.asset,
  ad_group_ad_asset_view.field_type,
  ad_group_ad_asset_view.performance_label,
  metrics.impressions,
  metrics.clicks,
  metrics.cost_micros,
  metrics.conversions
FROM ad_group_ad_asset_view
WHERE segments.date DURING LAST_MONTH
ORDER BY ad_group_ad_asset_view.performance_label
```

### Asset Source Information

- `Asset.source` is only accurate for mutable Assets
- For the source of RSA (Responsive Search Ad) Assets, use `AdGroupAdAsset.source`

## Best Practices Summary

1. Only select the fields you need to reduce response size and improve performance.
2. Apply filters in the `WHERE` clause to limit results to relevant data.
3. Before using a field in a query, check its metadata to ensure it's selectable, filterable, or sortable as needed.
4. Always use `ORDER BY` to ensure consistent results, especially when using pagination.
5. Use `LIMIT` for exploratory or display-only reports (e.g. 25–100). For full-data queries (all products, categorization, entire catalog), omit LIMIT — the backend paginates automatically.
6. For fields where `is_repeated = true`, use `CONTAINS ANY`, `CONTAINS ALL`, or `CONTAINS NONE`.
7. Be aware that including segment fields or certain attribute fields will cause metrics to be segmented in the results.
8. Use appropriate date functions and ranges for filtering by date segments.
9. For large result sets, use the page token provided in the response to retrieve subsequent pages.
10. For enum fields, verify the allowed values in the `enum_values` property before using them in queries.
