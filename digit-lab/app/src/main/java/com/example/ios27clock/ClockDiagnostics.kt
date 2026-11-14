package com.example.ios27clock

import android.content.Context
import android.os.Build
import android.os.Process
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.system.exitProcess

/** Digit Lab diagnostics persisted locally so render failures survive an activity restart. */
internal object ClockDiagnostics : RendererDiagnostics {
    private const val TAG = "LiquidClockDigitLab"
    private const val LOG_FILE_NAME = "digit-lab-diagnostics.log"
    private const val MAX_LOG_BYTES = 512 * 1024L

    private val lock = Any()
    @Volatile private var appContext: Context? = null

    fun initialize(context: Context) {
        appContext = context.applicationContext
        record("Digit Lab diagnostics initialized; SDK=" + Build.VERSION.SDK_INT)
    }

    fun installUncaughtExceptionHandler() {
        val current = Thread.getDefaultUncaughtExceptionHandler()
        if (current !is PersistingUncaughtExceptionHandler) {
            Thread.setDefaultUncaughtExceptionHandler(PersistingUncaughtExceptionHandler(current))
        }
    }

    override fun record(message: String) {
        Log.d(TAG, message)
        appendEvent("INFO", message)
    }

    override fun report(tag: String, throwable: Throwable) {
        Log.e(TAG, tag, throwable)
        val trace = StringWriter().also { writer -> throwable.printStackTrace(PrintWriter(writer)) }.toString()
        appendEvent("ERROR", tag + ": " + trace)
    }

    override fun reportMessage(title: String, details: String) {
        Log.e(TAG, title + ": " + details)
        appendEvent("ERROR", title + ": " + details)
    }

    fun createExportReport(context: Context, sessionDetails: String): String {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        val metrics = context.resources.displayMetrics
        val logContents = synchronized(lock) {
            val file = logFile(context)
            if (file.exists()) file.readText(Charsets.UTF_8) else "No persisted renderer events were recorded.\n"
        }
        return buildString {
            append("LIQUID CLOCK DIGIT LAB — DIAGNOSTIC EXPORT\n")
            append("Exported: ").append(timestamp()).append('\n')
            append("Capture scope: Digit Lab activity and shared renderer events; this is not a full Android system Logcat dump.\n\n")
            append("DEVICE\n")
            append("Manufacturer: ").append(Build.MANUFACTURER).append('\n')
            append("Model: ").append(Build.MODEL).append('\n')
            append("Device: ").append(Build.DEVICE).append('\n')
            append("Android release: ").append(Build.VERSION.RELEASE).append('\n')
            append("SDK: ").append(Build.VERSION.SDK_INT).append('\n')
            append("Build ID: ").append(Build.DISPLAY).append('\n')
            append("App ID: ").append(context.packageName).append('\n')
            append("App version: ").append(packageInfo.versionName ?: "unknown").append('\n')
            append("App version code: ").append(packageInfo.longVersionCode).append('\n')
            append("Display pixels: ").append(metrics.widthPixels).append('x').append(metrics.heightPixels).append('\n')
            append("Density: ").append(metrics.density).append(" (densityDpi=").append(metrics.densityDpi).append(")\n\n")
            append("CURRENT DIGIT LAB CONFIGURATION\n")
            append(sessionDetails).append("\n\n")
            append("PERSISTED DIAGNOSTIC EVENTS\n")
            append("Newest events are at the bottom. The log is capped at ").append(MAX_LOG_BYTES / 1024).append(" KiB.\n")
            append("------------------------------------------------------------\n")
            append(logContents)
        }
    }

    private fun appendEvent(level: String, details: String) {
        val context = appContext ?: return
        val entry = "[" + timestamp() + "][" + level + "] " + details.trimEnd() + "\n"
        val bytes = entry.toByteArray(Charsets.UTF_8)
        synchronized(lock) {
            try {
                val file = logFile(context)
                file.parentFile?.mkdirs()
                if (file.exists() && file.length() + bytes.size > MAX_LOG_BYTES) {
                    val previous = file.readBytes()
                    val tailStart = maxOf(0, previous.size - (MAX_LOG_BYTES / 2).toInt())
                    val tail = previous.copyOfRange(tailStart, previous.size)
                    val firstLineEnd = tail.indexOf('\n'.code.toByte())
                    val retained = if (firstLineEnd >= 0) tail.copyOfRange(firstLineEnd + 1, tail.size) else ByteArray(0)
                    file.writeBytes(retained)
                }
                file.appendBytes(bytes)
            } catch (exception: Exception) {
                Log.w(TAG, "Could not persist Digit Lab diagnostics.", exception)
            }
        }
    }

    private fun logFile(context: Context) = File(context.filesDir, LOG_FILE_NAME)

    private fun timestamp(): String =
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date())

    private class PersistingUncaughtExceptionHandler(
        private val previous: Thread.UncaughtExceptionHandler?
    ) : Thread.UncaughtExceptionHandler {
        override fun uncaughtException(thread: Thread, throwable: Throwable) {
            report("digitlab-uncaught-exception", throwable)
            val delegate = previous
            if (delegate != null && delegate !== this) {
                delegate.uncaughtException(thread, throwable)
            } else {
                Process.killProcess(Process.myPid())
                exitProcess(10)
            }
        }
    }
}
