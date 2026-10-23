// Run in spark-shell:  :load scala/DataQuality.scala
import org.apache.spark.sql.functions._

val df = spark.read.option("header", "true").option("inferSchema", "true")
  .csv("sample_telecom_churn.csv")

println(s"Rows: ${df.count()}  Columns: ${df.columns.length}")

// Missing values per column
df.select(df.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show()

// Duplicate rows
println(s"Duplicates: ${df.count() - df.dropDuplicates().count()}")

// Average charge and churn count per plan
df.groupBy("plan")
  .agg(avg("monthly_charge").alias("avg_charge"), count("*").alias("customers"))
  .orderBy(desc("customers")).show()
