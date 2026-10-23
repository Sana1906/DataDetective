import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

val inputPath = "sample_telecom_churn.csv"

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .option("mode", "PERMISSIVE")
  .csv(inputPath)

// 1. Dataset overview
val totalRows = df.count()
val totalColumns = df.columns.length

println("\n===== DATASET OVERVIEW =====")
println(s"Rows: $totalRows")
println(s"Columns: $totalColumns")

// 2. Column names and data types
println("\n===== COLUMN INFORMATION =====")
df.printSchema()

// 3. Missing values and missing percentages
println("\n===== MISSING VALUE REPORT =====")

val missingExpressions = df.columns.map { columnName =>
  val missing = col(columnName).isNull ||
    trim(col(columnName).cast("string")) === ""

  sum(when(missing, 1).otherwise(0))
    .cast("long")
    .alias(columnName)
}

val missingCounts = df.select(missingExpressions: _*).first()

df.columns.zipWithIndex.foreach { case (columnName, index) =>
  val missingCount = missingCounts.getLong(index)
  val percentage =
    if (totalRows == 0) 0.0
    else missingCount.toDouble / totalRows * 100

  println(
    f"$columnName%-25s Missing: $missingCount%d ($percentage%.2f%%)"
  )
}

// 4. Duplicate row detection
val distinctRows = df.dropDuplicates().count()
val duplicateRows = totalRows - distinctRows

println("\n===== DUPLICATE REPORT =====")
println(s"Duplicate rows: $duplicateRows")

// 5. Numerical descriptive statistics
val numericColumns = df.schema.fields.collect {
  case field if field.dataType.isInstanceOf[NumericType] =>
    field.name
}

println("\n===== NUMERICAL STATISTICS =====")

if (numericColumns.nonEmpty) {
  df.select(numericColumns.map(col): _*)
    .describe()
    .show(truncate = false)
} else {
  println("No numerical columns found.")
}

// 6. Data quality summary
val missingTotal = df.columns.indices.map { index =>
  missingCounts.getLong(index)
}.sum

val missingCells = df.columns.length.toLong * totalRows
val missingPercentage =
  if (missingCells == 0) 0.0
  else missingTotal.toDouble / missingCells * 100

println("\n===== DATA QUALITY SUMMARY =====")
println(s"Total missing cells: $missingTotal")
println(s"Missing cell percentage: ${"%.2f".format(missingPercentage)}%")
println(s"Duplicate rows: $duplicateRows")

if (missingTotal == 0 && duplicateRows == 0) {
  println("No missing or duplicate values detected.")
} else {
  println("Data cleaning is recommended before modelling.")
}
