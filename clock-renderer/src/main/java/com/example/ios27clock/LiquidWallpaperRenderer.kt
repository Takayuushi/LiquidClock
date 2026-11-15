package com.example.ios27clock

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.CornerPathEffect
import android.graphics.ColorFilter
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffColorFilter
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.RuntimeShader
import android.graphics.Shader
import android.graphics.Typeface
import android.util.Log
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Full-screen wallpaper clock compositor.
 *
 * Composition order:
 * 1. Original selected wallpaper
 * 2. Liquid-glass clock, clipped by the real glyph alpha mask
 * 3. The selected-object mask erases only clock pixels beneath the object
 */
class LiquidWallpaperRenderer(
    private val context: Context,
    private val wallpaperContent: RendererWallpaperContent = RendererWallpaperContent.PREVIEW,
    private val diagnostics: RendererDiagnostics = LogRendererDiagnostics,
    private val clockTextSource: RendererClockTextSource = SystemRendererClockTextSource
) {

    companion object {
        private const val TAG = "LiquidClockRenderer"
        private const val FONT_ASSET_PATH = "fonts/sf_pro_display_thin.otf"
        private const val VARIABLE_FONT_ASSET_PATH = "fonts/SF-Pro.ttf"

        private const val DEFAULT_CLOCK_X = 0.50f
        private const val DEFAULT_CLOCK_Y = 0.27f
        private const val LEGACY_DEFAULT_CLOCK_Y = 0.36f
        private const val PREVIOUS_DEFAULT_CLOCK_Y = 0.31f
        private const val MAX_GLYPH_HEIGHT_FRACTION = 0.38f
        private const val MAX_COMMITTED_HEIGHT_FRACTION = 0.74f
        private const val MAX_GLYPH_WIDTH_FRACTION = 0.94f
        private const val SF_PRO_WIDTH_COMPRESSED = 50
        private const val SF_PRO_WIDTH_CONDENSED = 60
        private const val SF_PRO_WIDTH_REGULAR = 100
        private const val SF_PRO_WIDTH_EXPANDED = 132
        private const val CLOCK_GLYPH_WIDTH_SCALE = 1.00f
        private const val CLOCK_GLYPH_HEIGHT_SCALE = 1.04f

        // Keep the natural horizontal layout while the adaptive clock grows
        // vertically. Individual glyph profiles handle optical corrections.
        private const val MIN_GLYPH_WIDTH_SCALE = 1.00f
        private const val MAX_GLYPH_WIDTH_SCALE = 1.00f
        private const val MIN_GLYPH_HEIGHT_SCALE = 1.00f
        private const val MAX_GLYPH_HEIGHT_SCALE = 1.04f
        private const val TRACKING_EM = 0.035f
        private const val BASE_WEIGHT_MIN = 520
        private const val BASE_WEIGHT_MAX = 820
        private const val COLON_ADVANCE_EM = 0.105f
        private const val COLON_DOT_RADIUS_EM = 0.040f
        private const val GLYPH_STROKE_FRACTION = 0.0f
        private const val COLON_DOT_OFFSET_EM = 0.16f
        private const val MIN_CLOCK_SCALE = 0.72f
        private const val MAX_CLOCK_SCALE = 2.55f
        private const val MIN_CLOCK_WIDTH_SCALE = 0.70f
        private const val MAX_CLOCK_WIDTH_SCALE = 1.65f
        private const val MAX_COMMITTED_WIDTH_FRACTION = 0.94f
        private const val SDF_RANGE_PX = 24f
        private const val SDF_PADDING_PX = 28
        private const val SDF_CARDINAL_COST = 5
        private const val SDF_DIAGONAL_COST = 7

        private val AGSL = """
        uniform shader uWallpaper;
        uniform shader uReflectionEnvironment;
        uniform shader uGlyphMask;
        uniform shader uColonMask;
        uniform shader uGlyphSdf;
        uniform float2 uResolution;
        uniform float2 uClockCenter;
        uniform float2 uSdfOrigin;
        uniform float2 uColonOrigin;
        uniform float uSdfScale;
        uniform float2 uTilt;
        uniform float uSdfRange;
        uniform float uFrosted;
        uniform float uTintStrength;
        uniform float uTime;
        uniform float3 uAccentColor;
        uniform float3 uClockTint;

        // Map a 3D direction in the glass' local camera frame to the
        // separate 2D reflection-proxy texture. X is right, Y is up, Z points
        // toward the viewer. Longitude spans left/right; latitude spans top/
        // bottom. This is an angular environment lookup, not a wallpaper-plane
        // intersection. The portrait wallpaper supplies colour, not geometry.
        float2 reflectionDirectionToUv(float3 direction) {
            float3 d = normalize(direction);

            // Fold the full direction sphere onto a single horizontal
            // hemisphere so the portrait proxy's complete width is useful.
            // abs(z) makes the mapping continuous through the horizon instead
            // of creating a wrap seam when reflected rays turn away from view.
            float horizontalMagnitude = length(d.xz);
            float longitude = atan(d.x, max(abs(d.z), 0.0001));
            float longitudeCoverage = smoothstep(
                0.025,
                0.12,
                horizontalMagnitude
            );
            float longitudeUv = longitude * 0.31830988618 + 0.5;
            // Longitude is undefined at the poles; converge smoothly to the
            // centre column there so tiny normal changes cannot jump the sample.
            float u = mix(0.5, longitudeUv, longitudeCoverage);

            float latitude = asin(clamp(d.y, -1.0, 1.0));
            float v = 0.5 - latitude * 0.31830988618;
            return clamp(
                float2(u, v),
                float2(0.0005),
                float2(0.9995)
            );
        }

        // Trace one narrow in-scattering direction from an interior point
        // to the local rear-interface tangent plane, then refract it into the
        // air gap and onto the backing-wallpaper plane. A failed/invalid ray
        // returns the legacy projected-path lookup so edge/TIR cases stay
        // bounded rather than producing extreme UVs.
        float4 scatterRayToWallpaper(
            float3 pathPosition,
            float2 fallbackPosition,
            float3 scatterDirection,
            float3 rearPlanePoint,
            float3 rearPlaneNormal,
            float2 frontCorrection,
            float rearPathValidity,
            float glassPathLength
        ) {
            float planeDenominator = dot(
                rearPlaneNormal,
                scatterDirection
            );
            float safePlaneDenominator = min(
                planeDenominator,
                -0.15
            );
            float planeTravel = dot(
                rearPlaneNormal,
                rearPlanePoint - pathPosition
            ) / safePlaneDenominator;
            float planeTravelValidity =
                step(0.15, -planeDenominator) *
                step(0.0, planeTravel) *
                step(planeTravel, glassPathLength * 2.0 + 2.0);

            float3 rearHit = pathPosition +
                scatterDirection * max(planeTravel, 0.0);
            float incidentCosine = clamp(
                -dot(rearPlaneNormal, scatterDirection),
                0.0,
                1.0
            );
            const float SCATTER_GLASS_IOR = 1.52;
            float snellK = 1.0 -
                SCATTER_GLASS_IOR * SCATTER_GLASS_IOR *
                (1.0 - incidentCosine * incidentCosine);
            float snellValidity = smoothstep(0.0, 0.08, snellK);
            float3 exitRay = normalize(
                scatterDirection * SCATTER_GLASS_IOR +
                rearPlaneNormal * (
                    SCATTER_GLASS_IOR * incidentCosine -
                    sqrt(max(snellK, 0.0))
                )
            );
            float exitDirectionValidity = step(0.15, -exitRay.z);
            float sampleValidity = clamp(
                rearPathValidity *
                    planeTravelValidity *
                    snellValidity *
                    exitDirectionValidity,
                0.0,
                1.0
            );

            float interfaceF0 = pow(
                (SCATTER_GLASS_IOR - 1.0) /
                    (SCATTER_GLASS_IOR + 1.0),
                2.0
            );
            float interfaceFresnel = interfaceF0 +
                (1.0 - interfaceF0) *
                pow(1.0 - incidentCosine, 5.0);
            float interfaceTransmission =
                (1.0 - interfaceFresnel) * step(0.0, snellK);
            float incomingBeerTransmission =
                exp(-0.0056 * max(planeTravel, 0.0));

            const float SCATTER_BACKGROUND_GAP_PX = 13.0;
            float backgroundTravel =
                SCATTER_BACKGROUND_GAP_PX /
                max(-exitRay.z, 0.15);
            float2 mappedWallpaperPoint =
                rearHit.xy +
                frontCorrection +
                exitRay.xy * backgroundTravel;
            float2 lookupPoint = clamp(
                mix(
                    fallbackPosition,
                    mappedWallpaperPoint,
                    sampleValidity
                ),
                float2(0.0),
                uResolution
            );
            float sourceTransmission = mix(
                1.0,
                interfaceTransmission * incomingBeerTransmission,
                sampleValidity
            );
            return float4(
                lookupPoint,
                sourceTransmission,
                sampleValidity
            );
        }

        half4 main(float2 p) {
            float coverage = uGlyphMask.eval(p).a;
            if (coverage < 0.002) {
                return half4(0.0);
            }
            float colonCoverage = uColonMask.eval(p - uColonOrigin).a;
            float colonFlag = smoothstep(0.18, 0.82, colonCoverage);

            // The liquid-glass surface is a lens, not a raised text bevel.
            // A wider distance profile lets refraction travel through the
            // translucent face while its narrow Fresnel rim catches light.
            float2 sdfPoint = (p - uSdfOrigin) * uSdfScale;
            float sdf = uGlyphSdf.eval(sdfPoint).a;
            float signedDistance = (sdf - 0.5) * 2.0 * uSdfRange;
            // A slightly wider finite-difference baseline filters single-pixel
            // SDF stair-steps before they become visible as fractured refraction.
            float2 sampleStep = float2(4.25, 4.25) * uSdfScale;
            float sdfLeft = uGlyphSdf.eval(sdfPoint - float2(sampleStep.x, 0.0)).a;
            float sdfRight = uGlyphSdf.eval(sdfPoint + float2(sampleStep.x, 0.0)).a;
            float sdfTop = uGlyphSdf.eval(sdfPoint - float2(0.0, sampleStep.y)).a;
            float sdfBottom = uGlyphSdf.eval(sdfPoint + float2(0.0, sampleStep.y)).a;
            float2 gradient = float2(sdfRight - sdfLeft, sdfBottom - sdfTop);
            float gradientLength = length(gradient);
            float normalConfidence = smoothstep(0.006, 0.055, gradientLength);
            float2 normal = -gradient / max(gradientLength, 0.0005);
            float2 refractionNormal = normal * normalConfidence;

            float solidMode = clamp(uFrosted, 0.0, 1.0);
            float insideDistance = max(signedDistance, 0.0);

            // Glass has a gently curved optical face and a broader, softer
            // Fresnel transition. Solid keeps its already-approved tighter edge.
            float glassEdge = exp(-abs(signedDistance) / 8.0);
            float solidEdge = exp(-abs(signedDistance) / 3.4);
            float edgeBand = mix(glassEdge, solidEdge, solidMode);
            float glassLens = smoothstep(-2.0, 6.0, signedDistance) *
                (1.0 - smoothstep(54.0, 118.0, signedDistance));
            float glassRim = exp(-abs(signedDistance - 1.15) / 6.2);
            float solidBevel = 1.0 -
                smoothstep(0.8, 13.0, insideDistance);
            // Model each digit stroke as a rounded glass rod, not a flat
            // plate with an embossed perimeter. The SDF measures distance
            // inward from either side of the stroke; an elliptical cap turns
            // that distance into a continuous optical surface normal.
            // Explicit rounded height map derived from the signed-distance
            // field. The glyph stroke becomes a convex optical cap:
            // h(d) = sqrt(2*R*d - d*d), with a flat centre after one radius.
            // Its gradient is sampled from neighbouring SDF texels so the
            // normal, refraction and lighting all describe the same surface.
            // The colon uses a shallower lens cap in Liquid Glass only.
            // Solid/Frosted always retains the original 20px profile.
            // Keep the Glass digit silhouette at the original cap width.
            // The previous 24px cap made the bright shoulder look too broad,
            // especially around the SF Pro 4 and 7.
            float glassDigitRadius = 20.0;
            float glassProfileRadius = mix(
                glassDigitRadius,
                3.2,
                colonFlag * (1.0 - solidMode)
            );
            float profileDistance = clamp(
                insideDistance,
                0.0,
                glassProfileRadius
            );
            float glassHeight = sqrt(max(
                2.0 * glassProfileRadius * profileDistance -
                    profileDistance * profileDistance,
                0.0
            ));

            float distanceLeft = max(
                (sdfLeft - 0.5) * 2.0 * uSdfRange,
                0.0
            );
            float distanceRight = max(
                (sdfRight - 0.5) * 2.0 * uSdfRange,
                0.0
            );
            float distanceTop = max(
                (sdfTop - 0.5) * 2.0 * uSdfRange,
                0.0
            );
            float distanceBottom = max(
                (sdfBottom - 0.5) * 2.0 * uSdfRange,
                0.0
            );
            float heightLeftDistance = min(distanceLeft, glassProfileRadius);
            float heightRightDistance = min(distanceRight, glassProfileRadius);
            float heightTopDistance = min(distanceTop, glassProfileRadius);
            float heightBottomDistance = min(distanceBottom, glassProfileRadius);
            float heightLeft = sqrt(max(
                2.0 * glassProfileRadius * heightLeftDistance -
                    heightLeftDistance * heightLeftDistance,
                0.0
            ));
            float heightRight = sqrt(max(
                2.0 * glassProfileRadius * heightRightDistance -
                    heightRightDistance * heightRightDistance,
                0.0
            ));
            float heightTop = sqrt(max(
                2.0 * glassProfileRadius * heightTopDistance -
                    heightTopDistance * heightTopDistance,
                0.0
            ));
            float heightBottom = sqrt(max(
                2.0 * glassProfileRadius * heightBottomDistance -
                    heightBottomDistance * heightBottomDistance,
                0.0
            ));
            float heightGradientX = (heightRight - heightLeft) / 8.5;
            float heightGradientY = (heightBottom - heightTop) / 8.5;
            float glassProfileSlope = length(
                float2(heightGradientX, heightGradientY)
            );
            float glassProfileCoverage = 1.0 -
                smoothstep(
                    glassProfileRadius * 0.78,
                    glassProfileRadius,
                    insideDistance
                );
            float3 glassSurfaceNormal = normalize(float3(
                -heightGradientX,
                -heightGradientY,
                1.0
            ));
            float3 keyLight3D = normalize(float3(-0.48, -0.66, 0.58));
            float3 fillLight3D = normalize(float3(0.62, 0.46, 0.64));
            float3 viewDirection3D = float3(0.0, 0.0, 1.0);
            float3 keyHalfVector = normalize(keyLight3D + viewDirection3D);
            float3 fillHalfVector = normalize(fillLight3D + viewDirection3D);
            float keyDiffuse3D = max(dot(glassSurfaceNormal, keyLight3D), 0.0);
            float fillDiffuse3D = max(dot(glassSurfaceNormal, fillLight3D), 0.0);
            // Schlick Fresnel from the actual height-map normal. The face
            // stays transmissive at normal incidence and reflects strongly
            // only as the curved surface turns toward grazing angles.
            float viewCosine = clamp(glassSurfaceNormal.z, 0.0, 1.0);
            float glassVolumeFresnel = 0.04 +
                0.96 * pow(1.0 - viewCosine, 5.0);
            float horizontalShoulder = smoothstep(0.52, 0.88, abs(glassSurfaceNormal.y));
            // Gently relax the horizontal-shoulder attenuation so horizontal
            // glass surfaces retain more of their directional reflection.
            // The limiter remains active to prevent broad bars from blooming.
            float glassHorizontalHighlight = mix(1.0, 0.66, horizontalShoulder);
            // Keep the broad softbox reflection on the rounded shoulder.
            // A front-facing normal has little Fresnel curvature and must not
            // receive a white fill across the whole face.
            float glassAreaReflection = pow(
                max(dot(glassSurfaceNormal, keyHalfVector), 0.0),
                5.5
            ) * 0.46 * smoothstep(0.08, 0.48, glassProfileSlope) * glassHorizontalHighlight;
            float glassKeyGlint = pow(
                max(dot(glassSurfaceNormal, keyHalfVector), 0.0),
                42.0
            ) * 0.88 * glassHorizontalHighlight;
            float glassFillGlint = pow(
                max(dot(glassSurfaceNormal, fillHalfVector), 0.0),
                30.0
            ) * 0.56 * glassHorizontalHighlight;
            float glassSideShadow = glassProfileCoverage *
                max(-dot(glassSurfaceNormal.xy, keyLight3D.xy), 0.0);
            float outerLip = 1.0 -
                smoothstep(0.4, 5.5, max(-signedDistance, 0.0));
            float interiorVolume = smoothstep(
                1.0,
                mix(26.0, 13.0, solidMode),
                signedDistance
            );
            // The edge profile describes the curved shoulder, while
            // interiorVolume describes the glass thickness behind it.
            // Combine them so transmitted scattering reaches the centre of
            // wide strokes instead of fading out after the first 20 pixels.
            // This mask is glyph-independent and is used only by Glass passes.
            float glassEdgeVolume = smoothstep(
                0.12,
                0.66,
                glassProfileCoverage
            );
            float glassCoreVolume = interiorVolume * 0.90;
            float glassVolumeRegion = max(
                glassEdgeVolume,
                glassCoreVolume
            );

            float2 tilt = clamp(uTilt, float2(-1.0), float2(1.0));

            // Snell refraction through the convex height-map surface.
            // Incoming view ray travels toward the glyph; eta is air/glass.
            const float GLASS_IOR = 1.52;
            const float GLASS_THICKNESS_PX = 46.0;
            float eta = 1.0 / GLASS_IOR;
            float cosIncident = clamp(glassSurfaceNormal.z, 0.0, 1.0);
            float refractK = max(
                1.0 - eta * eta * (1.0 - cosIncident * cosIncident),
                0.0
            );
            float refractFactor = eta * cosIncident - sqrt(refractK);
            float3 refractedRay = float3(0.0, 0.0, -eta) +
                glassSurfaceNormal * refractFactor;
            float glassEdgePull = mix(
                1.0,
                1.34,
                smoothstep(0.10, 0.55, glassProfileSlope)
            );
            // Estimate the local curvature of the convex height profile
            // from the same four SDF-derived height samples used for its normal.
            // The refracted-ray map is p + thickness * ray(p); its local
            // Jacobian therefore responds to the height Hessian. Keep this
            // magnification bounded and fade it out across the flat face/rim.
            const float GLASS_HEIGHT_SAMPLE_STEP = 4.25;
            float heightCurvatureX =
                (heightRight - 2.0 * glassHeight + heightLeft) /
                (GLASS_HEIGHT_SAMPLE_STEP * GLASS_HEIGHT_SAMPLE_STEP);
            float heightCurvatureY =
                (heightBottom - 2.0 * glassHeight + heightTop) /
                (GLASS_HEIGHT_SAMPLE_STEP * GLASS_HEIGHT_SAMPLE_STEP);
            // Complete the local 2D curvature tensor with the mixed derivative.
            // Diagonal wallpaper bands exercise this cross-axis curvature; the
            // earlier diagonal-only Jacobian could not represent their coupled
            // horizontal/vertical ray convergence. Sample only the shared SDF
            // height field, and skip these extra taps for Solid/Frosted.
            float heightCurvatureXY = 0.0;
            if (solidMode < 0.5) {
                float sdfTopLeft = uGlyphSdf.eval(
                    sdfPoint - sampleStep
                ).a;
                float sdfTopRight = uGlyphSdf.eval(
                    sdfPoint + float2(sampleStep.x, -sampleStep.y)
                ).a;
                float sdfBottomLeft = uGlyphSdf.eval(
                    sdfPoint + float2(-sampleStep.x, sampleStep.y)
                ).a;
                float sdfBottomRight = uGlyphSdf.eval(
                    sdfPoint + sampleStep
                ).a;
                float heightTopLeftDistance = clamp(
                    max((sdfTopLeft - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float heightTopRightDistance = clamp(
                    max((sdfTopRight - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float heightBottomLeftDistance = clamp(
                    max((sdfBottomLeft - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float heightBottomRightDistance = clamp(
                    max((sdfBottomRight - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float heightTopLeft = sqrt(max(
                    2.0 * glassProfileRadius * heightTopLeftDistance -
                        heightTopLeftDistance * heightTopLeftDistance,
                    0.0
                ));
                float heightTopRight = sqrt(max(
                    2.0 * glassProfileRadius * heightTopRightDistance -
                        heightTopRightDistance * heightTopRightDistance,
                    0.0
                ));
                float heightBottomLeft = sqrt(max(
                    2.0 * glassProfileRadius * heightBottomLeftDistance -
                        heightBottomLeftDistance * heightBottomLeftDistance,
                    0.0
                ));
                float heightBottomRight = sqrt(max(
                    2.0 * glassProfileRadius * heightBottomRightDistance -
                        heightBottomRightDistance * heightBottomRightDistance,
                    0.0
                ));
                heightCurvatureXY = (
                    heightBottomRight - heightBottomLeft -
                    heightTopRight + heightTopLeft
                ) / (4.0 * GLASS_HEIGHT_SAMPLE_STEP * GLASS_HEIGHT_SAMPLE_STEP);
            }
            float refractiveCurvatureScale =
                GLASS_THICKNESS_PX * 0.342;
            float rayMapJacobianX = 1.0 +
                refractiveCurvatureScale * heightCurvatureX;
            float rayMapJacobianY = 1.0 +
                refractiveCurvatureScale * heightCurvatureY;
            float rayMapJacobianXY =
                refractiveCurvatureScale * heightCurvatureXY;
            float rayMapDeterminant = clamp(
                rayMapJacobianX * rayMapJacobianY -
                    rayMapJacobianXY * rayMapJacobianXY,
                0.88,
                1.12
            );
            float lensRegion = smoothstep(
                0.025,
                0.16,
                glassProfileSlope
            ) * glassProfileCoverage;
            float2 baseGlassRefractionOffset =
                refractedRay.xy * GLASS_THICKNESS_PX * glassEdgePull;
            // Use the complete local 2D ray-map Jacobian, not just its scalar
            // area determinant. Its inverse carries directional stretch and
            // cross-axis coupling into the sampled wallpaper coordinate. This
            // is what lets diagonal bands bend around a curved glass stroke
            // instead of receiving only a nearly uniform 2D scale adjustment.
            float2 anisotropicGlassRefractionOffset = float2(
                rayMapJacobianY * baseGlassRefractionOffset.x -
                    rayMapJacobianXY * baseGlassRefractionOffset.y,
                rayMapJacobianX * baseGlassRefractionOffset.y -
                    rayMapJacobianXY * baseGlassRefractionOffset.x
            ) / rayMapDeterminant;
            // Bound the extra lens action so it cannot create edge kinks or
            // unstable magnification. Directional shear is retained, while
            // the total ray displacement may grow by at most 14 percent.
            float baseRefractionLength = length(baseGlassRefractionOffset);
            float mappedRefractionLength = length(
                anisotropicGlassRefractionOffset
            );
            float refractionDisplacementLimit = baseRefractionLength * 1.14;
            anisotropicGlassRefractionOffset *= min(
                1.0,
                refractionDisplacementLimit /
                    max(mappedRefractionLength, 0.0001)
            );
            float2 glassRefractionOffset = mix(
                baseGlassRefractionOffset,
                anisotropicGlassRefractionOffset,
                lensRegion
            );
            float2 solidWarp =
                refractionNormal *
                    (1.5 + solidEdge * 6.0 + solidBevel * 1.1) +
                tilt * 0.68;
            float2 glassWarp = glassRefractionOffset;

            // Trace the refracted ray through the opposite curved side of the
            // rounded glass rod. The rear surface is the mirrored SDF cap:
            // zBack = -h(distance) - thickness. Two fixed-point updates locate
            // the ray/surface intersection without a screen-space mirror guess.
            // This is a two-interface curved-lens approximation, shared by
            // every glyph; Solid/Frosted skips the entire extra geometry path.
            float glassBackInterfaceBlend = 0.0;
            // Estimate the actual local front-to-back distance through the
            // rounded glass section. The front and rear caps each contribute
            // their local height; Snell's ray angle converts z thickness to
            // travel distance. A valid traced rear hit below supersedes this
            // symmetric-profile estimate.
            float glassNominalTravel =
                (GLASS_THICKNESS_PX + 2.0 * glassHeight) /
                max(-refractedRay.z, 0.15);
            float2 glassInternalPathEnd =
                p + refractedRay.xy * glassNominalTravel;
            float glassPathLength =
                (GLASS_THICKNESS_PX + 2.0 * glassHeight) *
                length(refractedRay) /
                max(-refractedRay.z, 0.15);
            // The central rear hit supplies one local tangent plane for the
            // symmetric in-scattering rays. Invalid rear hits keep the legacy
            // projected-path fallback active.
            float2 glassScatterRearPlanePoint = glassInternalPathEnd;
            float glassScatterRearHeight = glassHeight;
            float3 glassScatterRearNormal = float3(0.0, 0.0, 1.0);
            float glassScatterRearValidity = 0.0;
            if (solidMode < 0.5) {
                float rearTravel = (
                    GLASS_THICKNESS_PX + 2.0 * glassHeight
                ) / max(-refractedRay.z, 0.15);
                float2 rearPoint = clamp(
                    p + refractedRay.xy * rearTravel,
                    float2(0.0),
                    uResolution
                );
                float sdfRearFirst = uGlyphSdf.eval(
                    sdfPoint + (rearPoint - p) * uSdfScale
                ).a;
                float rearFirstDistance = clamp(
                    max((sdfRearFirst - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearFirstHeight = sqrt(max(
                    2.0 * glassProfileRadius * rearFirstDistance -
                        rearFirstDistance * rearFirstDistance,
                    0.0
                ));
                // Refine the rear intersection twice after the initial
                // estimate. The cap height changes with the sampled SDF, so a
                // single update can leave a visible positional residual on
                // steeply curved strokes and turn diagonal wallpaper bands
                // into discontinuous dark stripes.
                rearTravel = (
                    GLASS_THICKNESS_PX + glassHeight + rearFirstHeight
                ) / max(-refractedRay.z, 0.15);
                rearPoint = clamp(
                    p + refractedRay.xy * rearTravel,
                    float2(0.0),
                    uResolution
                );
                float sdfRearSecond = uGlyphSdf.eval(
                    sdfPoint + (rearPoint - p) * uSdfScale
                ).a;
                float rearSecondDistance = clamp(
                    max((sdfRearSecond - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearSecondHeight = sqrt(max(
                    2.0 * glassProfileRadius * rearSecondDistance -
                        rearSecondDistance * rearSecondDistance,
                    0.0
                ));
                float refinedRearTravel = (
                    GLASS_THICKNESS_PX + glassHeight + rearSecondHeight
                ) / max(-refractedRay.z, 0.15);
                rearTravel = mix(rearTravel, refinedRearTravel, 0.72);
                rearPoint = clamp(
                    p + refractedRay.xy * rearTravel,
                    float2(0.0),
                    uResolution
                );
                float2 rearSdfPoint =
                    sdfPoint + (rearPoint - p) * uSdfScale;
                float sdfRear = uGlyphSdf.eval(rearSdfPoint).a;
                float rearDistance = clamp(
                    max((sdfRear - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearHeight = sqrt(max(
                    2.0 * glassProfileRadius * rearDistance -
                        rearDistance * rearDistance,
                    0.0
                ));

                // The rear cap faces the opposite direction, so its normal's
                // in-plane components have the opposite sign to the front cap.
                float sdfRearLeft = uGlyphSdf.eval(
                    rearSdfPoint - float2(sampleStep.x, 0.0)
                ).a;
                float sdfRearRight = uGlyphSdf.eval(
                    rearSdfPoint + float2(sampleStep.x, 0.0)
                ).a;
                float sdfRearTop = uGlyphSdf.eval(
                    rearSdfPoint - float2(0.0, sampleStep.y)
                ).a;
                float sdfRearBottom = uGlyphSdf.eval(
                    rearSdfPoint + float2(0.0, sampleStep.y)
                ).a;
                float rearLeftDistance = clamp(
                    max((sdfRearLeft - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearRightDistance = clamp(
                    max((sdfRearRight - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearTopDistance = clamp(
                    max((sdfRearTop - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearBottomDistance = clamp(
                    max((sdfRearBottom - 0.5) * 2.0 * uSdfRange, 0.0),
                    0.0,
                    glassProfileRadius
                );
                float rearHeightLeft = sqrt(max(
                    2.0 * glassProfileRadius * rearLeftDistance -
                        rearLeftDistance * rearLeftDistance,
                    0.0
                ));
                float rearHeightRight = sqrt(max(
                    2.0 * glassProfileRadius * rearRightDistance -
                        rearRightDistance * rearRightDistance,
                    0.0
                ));
                float rearHeightTop = sqrt(max(
                    2.0 * glassProfileRadius * rearTopDistance -
                        rearTopDistance * rearTopDistance,
                    0.0
                ));
                float rearHeightBottom = sqrt(max(
                    2.0 * glassProfileRadius * rearBottomDistance -
                        rearBottomDistance * rearBottomDistance,
                    0.0
                ));
                float rearHeightGradientX =
                    (rearHeightRight - rearHeightLeft) / 8.5;
                float rearHeightGradientY =
                    (rearHeightBottom - rearHeightTop) / 8.5;
                float3 glassRearNormal = normalize(float3(
                    rearHeightGradientX,
                    rearHeightGradientY,
                    1.0
                ));

                // Retain the actual curved rear hit and local normal for the
                // two directional in-scattering rays evaluated below.
                glassScatterRearPlanePoint = rearPoint;
                glassScatterRearHeight = rearHeight;
                glassScatterRearNormal = glassRearNormal;

                // Apply glass-to-air Snell refraction at the curved rear cap.
                // Total internal reflection falls back to the known-good
                // front-only mapping instead of generating an unstable ray.
                float rearIncidentCosine = clamp(
                    -dot(glassRearNormal, refractedRay),
                    0.0,
                    1.0
                );
                float rearEta = GLASS_IOR;
                float rearSnellK = 1.0 -
                    rearEta * rearEta *
                    (1.0 - rearIncidentCosine * rearIncidentCosine);
                float rearSnellGate = smoothstep(0.0, 0.08, rearSnellK);
                float rearSqrtK = sqrt(max(rearSnellK, 0.0));
                float3 glassBackExitRay = normalize(
                    refractedRay * rearEta +
                    glassRearNormal * (
                        rearEta * rearIncidentCosine - rearSqrtK
                    )
                );
                float rearInterfaceF0 = pow(
                    (GLASS_IOR - 1.0) / (GLASS_IOR + 1.0),
                    2.0
                );
                float rearInterfaceFresnel = rearInterfaceF0 +
                    (1.0 - rearInterfaceF0) *
                    pow(1.0 - rearIncidentCosine, 5.0);
                float rearInterfaceTransmission =
                    (1.0 - rearInterfaceFresnel) *
                    step(0.0, rearSnellK);

                // Continue a short distance from the rear interface to the
                // wallpaper plane. Keep the already accepted front Jacobian
                // as a correction rather than using it in place of the rear
                // intersection or the rear-surface normal.
                const float GLASS_BACKGROUND_GAP_PX = 13.0;
                float backExitTravel =
                    GLASS_BACKGROUND_GAP_PX /
                    max(-glassBackExitRay.z, 0.15);
                float2 frontJacobianCorrection =
                    glassRefractionOffset - baseGlassRefractionOffset;
                float2 curvedTwoInterfaceOffset =
                    (rearPoint - p) +
                    frontJacobianCorrection +
                    glassBackExitRay.xy * backExitTravel;

                float rearSurfaceCoverage = smoothstep(
                    0.50,
                    0.60,
                    sdfRear
                );
                float rearPathValidity =
                    rearSurfaceCoverage *
                    rearSnellGate *
                    rearInterfaceTransmission;
                // Keep geometry validity separate from the central ray's
                // Fresnel throughput; each off-axis ray computes its own
                // rear-interface transmission in scatterRayToWallpaper().
                glassScatterRearValidity =
                    rearSurfaceCoverage * rearSnellGate;
                // Use the rear intersection for internal path integration
                // only to the extent that the SDF hit and glass-to-air exit
                // are valid. Otherwise retain the nominal front-to-rear ray.
                // This keeps the quadrature endpoint consistent with the
                // same validity blend already used for optical path length.
                glassInternalPathEnd = clamp(
                    mix(
                        glassInternalPathEnd,
                        rearPoint,
                        rearPathValidity
                    ),
                    float2(0.0),
                    uResolution
                );
                glassPathLength = mix(
                    glassPathLength,
                    rearTravel * length(refractedRay),
                    rearPathValidity
                );
                float curvedShoulderCoverage =
                    smoothstep(0.025, 0.16, glassProfileSlope) *
                    glassProfileCoverage;
                // Give the valid curved two-interface ray a stronger say in
                // the shared transmitted sample. Keep all rear-coverage,
                // curvature, Snell and Fresnel gates unchanged so invalid or
                // near-flat regions continue using the established front path.
                // Build #187 validates the stronger shared curved-exit contribution.
                glassBackInterfaceBlend = 0.90 *
                    rearSurfaceCoverage *
                    curvedShoulderCoverage *
                    rearSnellGate *
                    rearInterfaceTransmission;
                glassWarp = mix(
                    glassRefractionOffset,
                    curvedTwoInterfaceOffset,
                    glassBackInterfaceBlend
                );
            }

            glassWarp += tilt * 1.8;
            float2 warp = mix(glassWarp, solidWarp, solidMode);
            float2 samplePoint = clamp(
                p + warp,
                float2(0.0),
                uResolution
            );

            float3 center = uWallpaper.eval(samplePoint).rgb;
            // Use the same solved optical path for both attenuation and a
            // four-bin Beer-Lambert quadrature of the environment radiance.
            // The four path taps below replace four of the former eight blur
            // taps, keeping the wallpaper sample budget unchanged.
            const float GLASS_EXTINCTION_PER_PX = 0.0056;
            const float GLASS_SCATTERING_ALBEDO = 0.88;
            float glassOpticalDepth =
                GLASS_EXTINCTION_PER_PX * glassPathLength;
            float glassDirectTransmittance = exp(-glassOpticalDepth);
            float glassInScatterEnergy =
                (1.0 - glassDirectTransmittance) *
                GLASS_SCATTERING_ALBEDO;
            float pathQuadratureDenominator = max(
                1.0 - glassDirectTransmittance,
                0.0001
            );
            float pathTauQuarter = glassOpticalDepth * 0.25;
            float pathWeightA =
                (1.0 - exp(-pathTauQuarter)) /
                pathQuadratureDenominator;
            float pathWeightB =
                (exp(-pathTauQuarter) - exp(-2.0 * pathTauQuarter)) /
                pathQuadratureDenominator;
            float pathWeightC =
                (exp(-2.0 * pathTauQuarter) - exp(-3.0 * pathTauQuarter)) /
                pathQuadratureDenominator;
            float pathWeightD =
                (exp(-3.0 * pathTauQuarter) - glassDirectTransmittance) /
                pathQuadratureDenominator;

            // The outgoing reflected rays live in the front/viewer
            // hemisphere. The backing wallpaper is behind the glass and is
            // therefore not a plane these rays can physically intersect.
            // Sample a separately named angular reflection proxy instead.
            float3 incidentView = float3(0.0, 0.0, -1.0);
            float3 reflectedView = normalize(
                incidentView -
                    2.0 * dot(incidentView, glassSurfaceNormal) *
                        glassSurfaceNormal
            );
            float2 reflectionUv = reflectionDirectionToUv(reflectedView);
            float3 environmentReflection =
                uReflectionEnvironment.eval(reflectionUv * uResolution).rgb;

            float3 keyEnvironmentRay = normalize(reflect(
                -keyLight3D,
                glassSurfaceNormal
            ));
            float3 fillEnvironmentRay = normalize(reflect(
                -fillLight3D,
                glassSurfaceNormal
            ));
            float2 keyReflectionUv =
                reflectionDirectionToUv(keyEnvironmentRay);
            float2 fillReflectionUv =
                reflectionDirectionToUv(fillEnvironmentRay);
            float3 keyEnvironmentReflection =
                uReflectionEnvironment.eval(keyReflectionUv * uResolution).rgb;
            float3 fillEnvironmentReflection =
                uReflectionEnvironment.eval(fillReflectionUv * uResolution).rgb;
            float3 directionalEnvironmentReflection = mix(
                fillEnvironmentReflection,
                keyEnvironmentReflection,
                0.64
            );

            // Select coordinates/throughput for the existing eight sample
            // slots. Liquid Glass uses a four-depth by two-direction
            // single-scattering quadrature; Solid/Frosted keeps the exact
            // original eight coordinates and unit sample weights.
            float2 wallpaperSampleA;
            float2 wallpaperSampleB;
            float2 wallpaperSampleC;
            float2 wallpaperSampleD;
            float2 wallpaperSampleE;
            float2 wallpaperSampleF;
            float2 wallpaperSampleG;
            float2 wallpaperSampleH;
            float sampleTransmissionA = 1.0;
            float sampleTransmissionB = 1.0;
            float sampleTransmissionC = 1.0;
            float sampleTransmissionD = 1.0;
            float sampleTransmissionE = 1.0;
            float sampleTransmissionF = 1.0;
            float sampleTransmissionG = 1.0;
            float sampleTransmissionH = 1.0;
            if (solidMode < 0.5) {
                float3 glassPathDirection = normalize(refractedRay);

                // Build the angular basis in the local plane of incidence:
                // project the rear-surface normal onto the plane perpendicular
                // to the central internal ray. This steers the paired samples
                // across the curved stroke section instead of along its tangent.
                float3 planeOfIncidenceAxis =
                    glassScatterRearNormal -
                    dot(glassScatterRearNormal, glassPathDirection) *
                        glassPathDirection;
                float planeAxisLength = length(planeOfIncidenceAxis);
                planeOfIncidenceAxis = planeOfIncidenceAxis /
                    max(planeAxisLength, 0.0001);

                // At near-normal incidence the plane of incidence is undefined.
                // Retain a stable screen-space tangent fallback, projected
                // perpendicular to the central ray so both basis vectors are
                // valid directions in the same angular plane.
                float2 tangentFallback2 = float2(
                    -glassPathDirection.y,
                    glassPathDirection.x
                );
                float tangentFallbackLength = length(tangentFallback2);
                tangentFallback2 = tangentFallbackLength > 0.0001
                    ? tangentFallback2 / tangentFallbackLength
                    : float2(1.0, 0.0);
                float3 tangentFallbackAxis = normalize(
                    float3(tangentFallback2, 0.0) -
                    dot(float3(tangentFallback2, 0.0), glassPathDirection) *
                        glassPathDirection
                );

                // The pair is symmetric, so align the two basis signs before
                // blending; this avoids an arbitrary 180-degree flip as the
                // surface approaches normal incidence.
                if (dot(planeOfIncidenceAxis, tangentFallbackAxis) < 0.0) {
                    planeOfIncidenceAxis = -planeOfIncidenceAxis;
                }
                float planeAxisBlend = smoothstep(
                    0.015,
                    0.12,
                    planeAxisLength
                );
                float3 scatterAxis = normalize(mix(
                    tangentFallbackAxis,
                    planeOfIncidenceAxis,
                    planeAxisBlend
                ));

                // A wider 16-degree symmetric cone gives the four depth
                // samples a meaningful cross-section footprint while the
                // per-ray Snell/TIR checks bound the shoulder response.
                const float SCATTER_ANGLE_COS = 0.9612616959;
                const float SCATTER_ANGLE_SIN = 0.2756373558;
                float3 scatterDirectionA = normalize(
                    glassPathDirection * SCATTER_ANGLE_COS +
                    scatterAxis * SCATTER_ANGLE_SIN
                );
                float3 scatterDirectionB = normalize(
                    glassPathDirection * SCATTER_ANGLE_COS -
                    scatterAxis * SCATTER_ANGLE_SIN
                );

                float3 frontPathPosition = float3(p, glassHeight);
                float3 rearTangentPoint = float3(
                    glassScatterRearPlanePoint,
                    -GLASS_THICKNESS_PX - glassScatterRearHeight
                );
                float3 rearPathPosition = mix(
                    frontPathPosition,
                    rearTangentPoint,
                    glassScatterRearValidity
                );
                float2 frontCorrection =
                    glassRefractionOffset - baseGlassRefractionOffset +
                    tilt * 1.8;

                float3 pathPositionA = mix(
                    frontPathPosition,
                    rearPathPosition,
                    0.125
                );
                float3 pathPositionB = mix(
                    frontPathPosition,
                    rearPathPosition,
                    0.375
                );
                float3 pathPositionC = mix(
                    frontPathPosition,
                    rearPathPosition,
                    0.625
                );
                float3 pathPositionD = mix(
                    frontPathPosition,
                    rearPathPosition,
                    0.875
                );

                float2 fallbackA = mix(p, glassInternalPathEnd, 0.125);
                float2 fallbackB = mix(p, glassInternalPathEnd, 0.375);
                float2 fallbackC = mix(p, glassInternalPathEnd, 0.625);
                float2 fallbackD = mix(p, glassInternalPathEnd, 0.875);

                float4 lookupA = scatterRayToWallpaper(
                    pathPositionA, fallbackA, scatterDirectionA,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupB = scatterRayToWallpaper(
                    pathPositionA, fallbackA, scatterDirectionB,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupC = scatterRayToWallpaper(
                    pathPositionB, fallbackB, scatterDirectionA,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupD = scatterRayToWallpaper(
                    pathPositionB, fallbackB, scatterDirectionB,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupE = scatterRayToWallpaper(
                    pathPositionC, fallbackC, scatterDirectionA,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupF = scatterRayToWallpaper(
                    pathPositionC, fallbackC, scatterDirectionB,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupG = scatterRayToWallpaper(
                    pathPositionD, fallbackD, scatterDirectionA,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );
                float4 lookupH = scatterRayToWallpaper(
                    pathPositionD, fallbackD, scatterDirectionB,
                    rearTangentPoint, glassScatterRearNormal,
                    frontCorrection, glassScatterRearValidity, glassPathLength
                );

                wallpaperSampleA = lookupA.xy;
                wallpaperSampleB = lookupB.xy;
                wallpaperSampleC = lookupC.xy;
                wallpaperSampleD = lookupD.xy;
                wallpaperSampleE = lookupE.xy;
                wallpaperSampleF = lookupF.xy;
                wallpaperSampleG = lookupG.xy;
                wallpaperSampleH = lookupH.xy;
                sampleTransmissionA = lookupA.z;
                sampleTransmissionB = lookupB.z;
                sampleTransmissionC = lookupC.z;
                sampleTransmissionD = lookupD.z;
                sampleTransmissionE = lookupE.z;
                sampleTransmissionF = lookupF.z;
                sampleTransmissionG = lookupG.z;
                sampleTransmissionH = lookupH.z;
            } else {
                float blurRadius = mix(44.0, 38.0, solidMode);
                float2 blurDiagonal = float2(
                    blurRadius * 0.7071,
                    blurRadius * 0.7071
                );
                wallpaperSampleA = clamp(
                    samplePoint + float2(blurRadius, 0.0),
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleB = clamp(
                    samplePoint - float2(blurRadius, 0.0),
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleC = clamp(
                    samplePoint + float2(0.0, blurRadius),
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleD = clamp(
                    samplePoint - float2(0.0, blurRadius),
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleE = clamp(
                    samplePoint + blurDiagonal,
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleF = clamp(
                    samplePoint - blurDiagonal,
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleG = clamp(
                    samplePoint + float2(blurDiagonal.x, -blurDiagonal.y),
                    float2(0.0),
                    uResolution
                );
                wallpaperSampleH = clamp(
                    samplePoint + float2(-blurDiagonal.x, blurDiagonal.y),
                    float2(0.0),
                    uResolution
                );
            }

            // Exactly eight wallpaper evaluations are shared by both material
            // paths; only the selected coordinates and Beer/Fresnel throughput
            // differ for Liquid Glass.
            float3 blurA =
                uWallpaper.eval(wallpaperSampleA).rgb * sampleTransmissionA;
            float3 blurB =
                uWallpaper.eval(wallpaperSampleB).rgb * sampleTransmissionB;
            float3 blurC =
                uWallpaper.eval(wallpaperSampleC).rgb * sampleTransmissionC;
            float3 blurD =
                uWallpaper.eval(wallpaperSampleD).rgb * sampleTransmissionD;
            float3 blurE =
                uWallpaper.eval(wallpaperSampleE).rgb * sampleTransmissionE;
            float3 blurF =
                uWallpaper.eval(wallpaperSampleF).rgb * sampleTransmissionF;
            float3 blurG =
                uWallpaper.eval(wallpaperSampleG).rgb * sampleTransmissionG;
            float3 blurH =
                uWallpaper.eval(wallpaperSampleH).rgb * sampleTransmissionH;

            // Average the symmetric angular pair at each depth, then integrate
            // those four directional radiance estimates with the existing
            // normalized Beer-Lambert interval weights.
            float3 pathIntegratedLight =
                (blurA + blurB) * (0.5 * pathWeightA) +
                (blurC + blurD) * (0.5 * pathWeightB) +
                (blurE + blurF) * (0.5 * pathWeightC) +
                (blurG + blurH) * (0.5 * pathWeightD);
            float3 legacyFrostedEnvironment = (
                blurA + blurB + blurC + blurD +
                blurE + blurF + blurG + blurH
            ) * 0.125;
            // Retain the current optical-depth response for broad reflections.
            // The Liquid Glass in-scatter source itself is now the directional
            // Beer-Lambert quadrature, not a blend with unrelated exit-plane
            // diagonal taps. Solid/Frosted retains its original eight-sample
            // equal average and exact sample locations.
            float glassPathScatterRoughness = smoothstep(
                0.30,
                0.58,
                glassOpticalDepth
            );
            float3 frostedWallpaper = mix(
                legacyFrostedEnvironment,
                pathIntegratedLight,
                1.0 - solidMode
            );
            // Extract bright energy from the same directional path samples so
            // the bloom source remains spatially aligned with the in-scatter.
            float brightA = smoothstep(
                0.14, 0.52,
                dot(blurA, float3(0.2126, 0.7152, 0.0722))
            );
            float brightB = smoothstep(
                0.14, 0.52,
                dot(blurB, float3(0.2126, 0.7152, 0.0722))
            );
            float brightC = smoothstep(
                0.14, 0.52,
                dot(blurC, float3(0.2126, 0.7152, 0.0722))
            );
            float brightD = smoothstep(
                0.14, 0.52,
                dot(blurD, float3(0.2126, 0.7152, 0.0722))
            );
            float brightE = smoothstep(
                0.14, 0.52,
                dot(blurE, float3(0.2126, 0.7152, 0.0722))
            );
            float brightF = smoothstep(
                0.14, 0.52,
                dot(blurF, float3(0.2126, 0.7152, 0.0722))
            );
            float brightG = smoothstep(
                0.14, 0.52,
                dot(blurG, float3(0.2126, 0.7152, 0.0722))
            );
            float brightH = smoothstep(
                0.14, 0.52,
                dot(blurH, float3(0.2126, 0.7152, 0.0722))
            );
            float3 pathIntegratedBrightBloom =
                (blurA * brightA + blurB * brightB) *
                    (0.5 * pathWeightA) +
                (blurC * brightC + blurD * brightD) *
                    (0.5 * pathWeightB) +
                (blurE * brightE + blurF * brightF) *
                    (0.5 * pathWeightC) +
                (blurG * brightG + blurH * brightH) *
                    (0.5 * pathWeightD);
            // Preserve the exact original Solid/Frosted bloom path: its
            // diagonal source is blurE..H and its path source weights A..D
            // individually. Liquid Glass instead uses the paired 4x2
            // Beer-Lambert quadrature at the four depth midpoints.
            float3 solidPathIntegratedBrightBloom =
                blurA * brightA * pathWeightA +
                blurB * brightB * pathWeightB +
                blurC * brightC * pathWeightC +
                blurD * brightD * pathWeightD;
            float3 solidDiagonalBrightBloom = (
                blurE * brightE + blurF * brightF +
                blurG * brightG + blurH * brightH
            ) * 0.25;
            float3 solidLegacyBrightBloom = mix(
                solidDiagonalBrightBloom,
                solidPathIntegratedBrightBloom,
                0.36
            );
            float3 glassBrightBloom = mix(
                solidLegacyBrightBloom,
                pathIntegratedBrightBloom,
                1.0 - solidMode
            );
            // Keep the green channel on the central refracted ray. Red and
            // blue take tiny opposite offsets along the same optical path,
            // producing restrained chromatic dispersion rather than a broad
            // painted rainbow fringe.
            float dispersionAmount = mix(
                3.2,
                5.8,
                smoothstep(0.10, 0.48, glassProfileSlope)
            );
            float2 dispersionOffset = refractedRay.xy * dispersionAmount;
            float3 glassRedSample = uWallpaper.eval(clamp(
                samplePoint + dispersionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            float3 glassBlueSample = uWallpaper.eval(clamp(
                samplePoint - dispersionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            float3 refractedGlass = float3(
                glassRedSample.r,
                center.g,
                glassBlueSample.b
            );

            // Directional microfacet transmission: estimate the local wallpaper
            // edge normal, then diffuse ACROSS high-contrast bands rather than
            // along them. This merges paired white streaks into a coherent
            // optical body while retaining their direction and larger detail.
            // It is one image-driven material response shared by every glyph.
            // A slightly broader neighbourhood stabilises the cross-band
            // orientation estimate where several narrow parallel streaks
            // sit close together. This changes direction estimation only;
            // transmission radii and blend weights remain untouched.
            float2 diffusionProbe = float2(5.5, 5.5);
            float3 probeLeft = uWallpaper.eval(clamp(
                samplePoint - float2(diffusionProbe.x, 0.0),
                float2(0.0),
                uResolution
            )).rgb;
            float3 probeRight = uWallpaper.eval(clamp(
                samplePoint + float2(diffusionProbe.x, 0.0),
                float2(0.0),
                uResolution
            )).rgb;
            float3 probeTop = uWallpaper.eval(clamp(
                samplePoint - float2(0.0, diffusionProbe.y),
                float2(0.0),
                uResolution
            )).rgb;
            float3 probeBottom = uWallpaper.eval(clamp(
                samplePoint + float2(0.0, diffusionProbe.y),
                float2(0.0),
                uResolution
            )).rgb;
            // A full-RGB structure tensor finds the dominant local colour-edge
            // normal. Unlike luminance-gradient signs, its eigenvector remains
            // stable on saturated blue/black transitions and mixed-colour bands.
            // The principal direction is the cross-band sampling axis.
            float3 colourGradientX = probeRight - probeLeft;
            float3 colourGradientY = probeBottom - probeTop;
            float tensorXX = dot(colourGradientX, colourGradientX);
            float tensorYY = dot(colourGradientY, colourGradientY);
            float tensorXY = dot(colourGradientX, colourGradientY);
            float tensorDiscriminant = sqrt(max(
                (tensorXX - tensorYY) * (tensorXX - tensorYY) +
                    4.0 * tensorXY * tensorXY,
                0.0
            ));
            float tensorLambda = 0.5 *
                (tensorXX + tensorYY + tensorDiscriminant);
            float2 tensorVectorX = float2(
                tensorXY,
                tensorLambda - tensorXX
            );
            float2 tensorVectorY = float2(
                tensorLambda - tensorYY,
                tensorXY
            );
            float2 diffusionDirection = mix(
                tensorVectorX,
                tensorVectorY,
                step(tensorYY, tensorXX)
            );
            float diffusionGradientLength = length(diffusionDirection);
            float2 diffusionAxis = diffusionDirection /
                max(diffusionGradientLength, 0.0001);
            float wallpaperContrast = sqrt(max(tensorLambda, 0.0));
            // Let moderate-contrast streaks enter the same cross-band blending
            // response a little earlier. Keep the maximum diffusion footprint
            // fixed so already-strongly-blended regions (such as the orange/
            // white transition in the zero) are not pushed into extra haze.
            float bandContrast = smoothstep(0.06, 0.45, wallpaperContrast);
            // Diffuse across (not along) high-contrast wallpaper bands.
            // The larger footprint merges paired streaks into a continuous
            // glass volume, while low-contrast areas retain sharper detail.
            // Slightly widen only the across-band footprint where the wallpaper
            // tensor detects strong parallel streaks. This targets residual
            // dark-line separation without increasing the whole-volume haze.
            // Let the same cross-band diffusion footprint grow with the
            // actual Beer-Lambert path through the rounded glass volume.
            // Thin shoulders keep the sharper refraction; thicker interior
            // paths receive a modestly broader neighbourhood, creating a
            // softer, more cohesive internal blend without a global blur.
            float glassDiffusionDepth = smoothstep(
                0.18,
                0.52,
                glassOpticalDepth
            );
            float diffusionRadius = mix(24.0, 40.0, bandContrast) *
                mix(1.0, 1.16, glassDiffusionDepth);
            float2 nearDiffusionOffset = diffusionAxis * diffusionRadius * 0.62;
            float2 farDiffusionOffset = diffusionAxis * diffusionRadius * 1.38;
            float3 softNearA = uWallpaper.eval(clamp(
                samplePoint - nearDiffusionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            float3 softNearB = uWallpaper.eval(clamp(
                samplePoint + nearDiffusionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            float3 softFarA = uWallpaper.eval(clamp(
                samplePoint - farDiffusionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            float3 softFarB = uWallpaper.eval(clamp(
                samplePoint + farDiffusionOffset,
                float2(0.0),
                uResolution
            )).rgb;
            // A tiny redistribution toward the wider pair adds a little
            // more low-frequency colour merging without increasing blur radius
            // or changing total transmitted energy.
            float3 microfacetTransmission =
                center * 0.06 +
                (softNearA + softNearB) * 0.24 +
                (softFarA + softFarB) * 0.23;

            // Stabilise paired high-contrast rails: when the local RGB tensor
            // has an ambiguous direction, or the wallpaper contains strong
            // adjacent bands, blend a restrained amount of the existing
            // colour-preserving broad environment into the directional bundle.
            // This is shared per-pixel transport, not a glyph-specific rule.
            float tensorTrace = max(tensorXX + tensorYY, 0.0001);
            float tensorCoherence = clamp(
                tensorDiscriminant / tensorTrace,
                0.0,
                1.0
            );
            float directionalConfidence = smoothstep(
                0.12,
                0.55,
                tensorCoherence
            );
            float3 isotropicFallbackTransmission = mix(
                center,
                frostedWallpaper,
                0.72
            );
            float pairedBandSoftening = mix(
                1.0,
                0.72,
                bandContrast
            );
            float stableDirectionWeight =
                mix(0.82, 1.0, directionalConfidence) *
                pairedBandSoftening;
            microfacetTransmission = mix(
                isotropicFallbackTransmission,
                microfacetTransmission,
                stableDirectionWeight
            );

            // Preserve warm wallpaper energy only when the final, coherence-
            // stabilised transmission genuinely contains red/amber chroma.
            // Work in linear light so the correction follows optical energy,
            // not display-encoded channel values. Neutral white is rejected
            // by requiring red to exceed both green and blue.
            float3 microfacetTransmissionLinearHigh = float3(
                pow((microfacetTransmission.r + 0.055) / 1.055, 2.4),
                pow((microfacetTransmission.g + 0.055) / 1.055, 2.4),
                pow((microfacetTransmission.b + 0.055) / 1.055, 2.4)
            );
            float3 microfacetTransmissionLinear = mix(
                microfacetTransmission / 12.92,
                microfacetTransmissionLinearHigh,
                step(float3(0.04045), microfacetTransmission)
            );
            float warmLightSignal = max(
                microfacetTransmissionLinear.r -
                    max(
                        microfacetTransmissionLinear.g * 0.98,
                        microfacetTransmissionLinear.b * 1.04
                    ),
                0.0
            );
            float warmLightPresence = smoothstep(
                0.025,
                0.12,
                warmLightSignal
            ) * smoothstep(
                0.045,
                0.22,
                microfacetTransmissionLinear.r
            );
            float3 warmLightTransportLinear =
                microfacetTransmissionLinear *
                    (1.0 + warmLightPresence * 0.16) +
                float3(0.010, 0.004, 0.0) * warmLightPresence;
            float3 warmLightTransportSrgbHigh = float3(
                1.055 * pow(max(warmLightTransportLinear.r, 0.0), 0.4166667) - 0.055,
                1.055 * pow(max(warmLightTransportLinear.g, 0.0), 0.4166667) - 0.055,
                1.055 * pow(max(warmLightTransportLinear.b, 0.0), 0.4166667) - 0.055
            );
            float3 warmLightTransport = mix(
                warmLightTransportLinear * 12.92,
                warmLightTransportSrgbHigh,
                step(
                    float3(0.0031308),
                    warmLightTransportLinear
                )
            );

            float microfacetDiffusion = mix(
                0.42,
                0.96,
                bandContrast
            ) * mix(
                0.96,
                1.0,
                smoothstep(0.04, 0.32, glassProfileSlope)
            );
            // Preserve the established cross-band luminance diffusion,
            // but do not force the same amount of chroma averaging onto the
            // direct refracted/dispersion source. This keeps environmental
            // colour carried by the actual refracted ray while the broad
            // microfacet source still controls the soft internal luminance.
            float preScatterRefractedLuma = dot(
                refractedGlass,
                float3(0.2126, 0.7152, 0.0722)
            );
            float preScatterMicrofacetLuma = dot(
                microfacetTransmission,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 preScatterRefractedChroma =
                refractedGlass - float3(preScatterRefractedLuma);
            float3 preScatterMicrofacetChroma =
                microfacetTransmission - float3(preScatterMicrofacetLuma);
            float preScatterChromaSeparation = length(
                preScatterRefractedChroma -
                    preScatterMicrofacetChroma
            );
            float preScatterChromaRetention = smoothstep(
                0.025,
                0.16,
                preScatterChromaSeparation
            );
            float preScatterLuma = mix(
                preScatterRefractedLuma,
                preScatterMicrofacetLuma,
                microfacetDiffusion
            );
            // Retain up to sixteen additional percentage points of the original
            // refracted-source chroma where the two existing sources differ.
            // This applies equally to all hues and glyphs; no warm/blue/cyan
            // detector or fixed tint is introduced.
            float preScatterChromaDiffusion = clamp(
                microfacetDiffusion -
                    0.16 * preScatterChromaRetention,
                0.0,
                1.0
            );
            float3 preScatterChroma = mix(
                preScatterRefractedChroma,
                preScatterMicrofacetChroma,
                preScatterChromaDiffusion
            );
            // Keep the reconstructed RGB in gamut without changing its
            // luminance: scale chroma uniformly only when required by a
            // channel boundary.
            float3 positiveChroma = max(preScatterChroma, float3(0.0));
            float3 negativeChroma = max(-preScatterChroma, float3(0.0));
            float3 positiveChromaCapacity =
                (1.0 - preScatterLuma) /
                max(positiveChroma, float3(0.0001));
            float3 negativeChromaCapacity =
                preScatterLuma /
                max(negativeChroma, float3(0.0001));
            float preScatterGamutScale = clamp(
                min(
                    min(
                        min(positiveChromaCapacity.x, positiveChromaCapacity.y),
                        positiveChromaCapacity.z
                    ),
                    min(
                        min(negativeChromaCapacity.x, negativeChromaCapacity.y),
                        negativeChromaCapacity.z
                    )
                ),
                0.0,
                1.0
            );
            refractedGlass = float3(preScatterLuma) +
                preScatterChroma * preScatterGamutScale;

            float refractedLuma = dot(
                refractedGlass,
                float3(0.2126, 0.7152, 0.0722)
            );
            // Retain saturation in ordinary areas, but do not re-amplify
            // high-contrast wallpaper bands after the common diffusion pass.
            // The adjustment is image-driven and identical for every glyph.
            float glassTransmissionSaturation = mix(
                1.42,
                1.08,
                bandContrast
            );
            float3 saturatedRefractedGlass = clamp(
                mix(
                    float3(refractedLuma),
                    refractedGlass,
                    glassTransmissionSaturation
                ),
                0.0,
                1.0
            );
            // Rich internal light pickup: dark regions gather nearby colour
            // from the wallpaper, while bright beach/sky regions remain clear.
            // Requiring local blurred illumination avoids a milky veil where
            // the surrounding wallpaper is already uniformly bright.
            float blurredWallpaperLuma = dot(
                frostedWallpaper,
                float3(0.2126, 0.7152, 0.0722)
            );
            float glassDarkness = 1.0 - smoothstep(0.08, 0.24, refractedLuma);
            float glassNearbyLight = smoothstep(0.10, 0.32, blurredWallpaperLuma);
            float glassBounceStrength =
                0.24 * glassDarkness * glassNearbyLight;
            float3 brightGlassBounce = clamp(
                frostedWallpaper * 1.44 + float3(0.012, 0.014, 0.022),
                0.0,
                1.0
            );
            float3 screenedGlassFace = 1.0 -
                (1.0 - saturatedRefractedGlass) *
                (1.0 - brightGlassBounce);
            float3 glassTransmitted = mix(
                saturatedRefractedGlass,
                screenedGlassFace,
                glassBounceStrength
            );
            // Treat transmission as one glass medium: a small directional
            // microfacet component preserves optical refraction, while the
            // isotropic wallpaper environment supplies the broad atmospheric
            // in-scattering seen through hazy illuminated glass.
            float3 warmRetainedTransmission = mix(
                microfacetTransmission,
                warmLightTransport,
                warmLightPresence * 0.42
            );
            // Transmission now passes through the curved rear interface
            // above. Do not add the old parallel-slab Fresnel return samples:
            // they boosted local colour without placing a physically correct
            // opposite-side image and were mostly erased by broad scattering.
            float3 volumeScatteredTransmission = clamp(
                warmRetainedTransmission * 1.10 +
                    float3(0.016, 0.020, 0.028),
                0.0,
                1.0
            );
            float glassAtmosphericCoverage = smoothstep(
                0.10,
                0.60,
                glassVolumeRegion
            );
            // The convex profile and traced rear interface define a shared
            // relative optical path through every glyph. Convert that path to
            // a physical pixel length, then use Beer-Lambert attenuation to
            // split the transmitted radiance into a direct lobe and a broad
            // in-scattered lobe. The removed direct energy is not independently
            // averaged into both luma and chroma and then mixed a second time.
            // Extinction is calibrated so the full 86px centre thickness
            // retains approximately the same normal-incidence attenuation
            // as the prior 46px nominal-path model. Thinner shoulders therefore
            // scatter less; longer oblique/rear-traced paths scatter more.
            // Keep the existing image-adaptive chroma and band response, but
            // combine the directional transmission and broad environment into
            // one RGB in-scatter source. The broad lobe retains the local
            // refracted colour rather than separately averaging most luma and
            // chroma toward the blurred wallpaper.
            float volumeTransmissionLuma = dot(
                volumeScatteredTransmission,
                float3(0.2126, 0.7152, 0.0722)
            );
            float frostedWallpaperLuma = dot(
                frostedWallpaper,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 volumeTransmissionChroma =
                volumeScatteredTransmission - float3(volumeTransmissionLuma);
            float3 frostedWallpaperChroma =
                frostedWallpaper - float3(frostedWallpaperLuma);
            float chromaticSeparation = length(
                volumeTransmissionChroma - frostedWallpaperChroma
            );
            float chromaticRetention = smoothstep(
                0.025,
                0.16,
                chromaticSeparation
            );
            // Preserve directional colour where the curved rear path is valid
            // and the shared surface normal is strongly horizontal-facing.
            // This remains a small geometry-conditioned adjustment, not a
            // glyph-specific or colour-specific branch.
            float glassCurvedShoulderColourTransport =
                glassBackInterfaceBlend * horizontalShoulder;
            float glassCurvatureDirectionalRetention =
                0.035 * glassCurvedShoulderColourTransport;
            // Give the broad, path-integrated environment a slightly
            // larger share of the in-scattered lobe. Beer-Lambert extinction
            // still controls how much energy enters this lobe; the local
            // structure tensor only controls how its broad share responds to
            // the wallpaper's band contrast. This softens the transmitted
            // image by mixing nearby radiance through the same glass volume,
            // without a screen-space blur or a colour-specific correction.
            float glassVolumeIsotropicWeight = clamp(
                mix(0.70, 0.76, bandContrast) -
                    0.04 * chromaticRetention -
                    glassCurvatureDirectionalRetention,
                0.62,
                0.76
            );
            // Keep the established broad luminance diffusion exactly as-is,
            // but let more directional environmental chroma survive when the
            // two existing sources carry meaningfully different colours.
            // This is hue-agnostic: every channel uses the same chroma weight,
            // and the colour can only come from the sampled wallpaper.
            float glassVolumeLumaIsotropicWeight =
                glassVolumeIsotropicWeight;
            // The broad lobe is attenuated again by Beer-Lambert scatter
            // energy and atmospheric coverage after this source blend. Scale
            // the source-level chroma retention by that downstream weight so
            // its maximum final contribution is comparable to the earlier
            // #136 2.5%-of-output chroma-only refinement, rather than being
            // reduced to roughly 1.3% at the nominal 86px optical path.
            // The 0.08 cap prevents thin paths from receiving an excessive
            // source-level adjustment; luminance and energy remain unchanged.
            float glassChromaOutputRetentionBudget = min(
                0.08,
                0.025 / max(
                    glassInScatterEnergy * glassAtmosphericCoverage,
                    0.3125
                )
            );
            float glassVolumeChromaIsotropicWeight = clamp(
                glassVolumeIsotropicWeight -
                    glassChromaOutputRetentionBudget *
                        chromaticRetention,
                0.58,
                0.76
            );
            float glassBroadInScatterLuma = mix(
                volumeTransmissionLuma,
                frostedWallpaperLuma,
                glassVolumeLumaIsotropicWeight
            );
            float3 glassBroadInScatterChroma = mix(
                volumeTransmissionChroma,
                frostedWallpaperChroma,
                glassVolumeChromaIsotropicWeight
            );
            float3 glassBroadInScatter = clamp(
                float3(glassBroadInScatterLuma) +
                    glassBroadInScatterChroma,
                0.0,
                1.0
            );

            // Two-lobe transport: attenuated direct refracted radiance plus
            // the corresponding fraction of broad, colour-bearing in-scatter.
            // The weights are energy-consistent: direct T plus scattered
            // (1-T)*albedo, with the remainder representing absorption.
            float3 glassTwoLobeTransport = clamp(
                glassTransmitted * glassDirectTransmittance +
                    glassBroadInScatter * glassInScatterEnergy,
                0.0,
                1.0
            );
            glassTransmitted = mix(
                glassTransmitted,
                glassTwoLobeTransport,
                glassAtmosphericCoverage
            );
            // Bright-pass bloom is screen-composited at low strength so
            // illuminated regions spread softly without turning the complete
            // face into a white or gray frosted layer.
            float glassAtmosphericBloom =
                0.30 * glassAtmosphericCoverage;
            glassTransmitted = 1.0 -
                (1.0 - glassTransmitted) *
                (1.0 - glassBrightBloom * glassAtmosphericBloom);
            // Solid keeps its prior diffuse/frosted source path.
            float3 refracted = mix(glassTransmitted, center, solidMode);
            float3 milkyFrost = mix(
                frostedWallpaper,
                float3(0.94, 0.95, 0.98),
                0.20
            );
            milkyFrost = mix(milkyFrost, uAccentColor, 0.07);

            // Clear glass still reflects its environment. Mix a restrained
            // wallpaper-coloured reflection into the face, then increase it
            // toward the curved shoulder using Fresnel. This gives the front
            // surface a visible optical body without a white/frosted overlay.
            float environmentReflectionLuma = dot(
                environmentReflection,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 environmentReflectionChroma =
                environmentReflection - float3(environmentReflectionLuma);
            float reflectionHighlightProtection = 1.0 -
                0.58 * smoothstep(0.42, 0.78, environmentReflectionLuma);
            float3 chromaticEnvironmentReflection = clamp(
                float3(environmentReflectionLuma) +
                    environmentReflectionChroma * 1.28,
                0.0,
                1.0
            );
            float glassFaceReflectivity =
                glassVolumeFresnel * 0.64 * reflectionHighlightProtection;
            // Convolve a portion of the environment reflection with the same
            // local wallpaper volume. The broad reflection blends into the
            // transmitted haze while the later key/fill glints preserve crisp
            // glass contours and specular definition.
            // Let the broad reflected environment follow both the
            // convex surface and the actual Beer-Lambert path through the
            // glass. Thick, strongly scattering regions receive a softer
            // reflected environment; thin regions retain a clearer image.
            // This couples surface reflection to internal transport without
            // increasing reflection energy or adding another wallpaper tap.
            float glassSurfaceCurvatureRoughness = smoothstep(
                0.08,
                0.48,
                glassProfileSlope
            );
            float glassReflectionRoughness = mix(
                0.22,
                0.30,
                max(
                    glassSurfaceCurvatureRoughness,
                    glassPathScatterRoughness
                )
            );
            float3 glassSoftEnvironmentReflection = mix(
                chromaticEnvironmentReflection,
                frostedWallpaper,
                glassReflectionRoughness
            );
            // Compose reflected luminance and chroma separately. The reflection
            // keeps its full brightness response, while a small portion of the
            // refracted wallpaper chroma remains visible beneath it. This makes
            // transmission and reflection read as one coloured glass volume.
            float refractedFaceLuma = dot(
                refracted,
                float3(0.2126, 0.7152, 0.0722)
            );
            float reflectedFaceLuma = dot(
                glassSoftEnvironmentReflection,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 refractedFaceChroma =
                refracted - float3(refractedFaceLuma);
            float3 reflectedFaceChroma =
                glassSoftEnvironmentReflection - float3(reflectedFaceLuma);
            float transmittedFaceLuma = mix(
                refractedFaceLuma,
                reflectedFaceLuma,
                glassFaceReflectivity
            );
            // Retain a small amount of reflected environment colour across
            // front-facing glass, where Fresnel alone approaches its minimum.
            // Keep this floor chroma-only: the transmitted face luminance is
            // untouched, so dark glass stays dark instead of becoming a haze.
            float glassFaceChromaReflectivity = max(
                glassFaceReflectivity * 0.78,
                0.055 * reflectionHighlightProtection
            );
            float3 integratedFaceChroma = mix(
                refractedFaceChroma,
                reflectedFaceChroma,
                glassFaceChromaReflectivity
            );
            float3 glassFace = clamp(
                float3(transmittedFaceLuma) + integratedFaceChroma,
                0.0,
                1.0
            );
            float innerGlassTint = smoothstep(0.0, 12.0, insideDistance) *
                (1.0 - smoothstep(26.0, 64.0, insideDistance));
            glassFace = mix(
                glassFace,
                mix(uAccentColor, glassFace, 0.78),
                innerGlassTint * 0.08
            );
            glassFace = mix(glassFace, uClockTint, uTintStrength);
            float3 frostedFace = mix(
                milkyFrost,
                uClockTint,
                uTintStrength * 0.78
            );
            float3 body = mix(glassFace, frostedFace, solidMode);

            float2 keyLight = normalize(float2(-0.58, -0.78) + tilt * 0.62);
            float2 fillLight = normalize(float2(0.70, 0.64) - tilt * 0.24);
            float facing = max(dot(normal, keyLight), 0.0);
            float oppositeFacing = max(dot(normal, fillLight), 0.0);

            float legacyFresnel = pow(
                1.0 - clamp(abs(dot(normal, keyLight)), 0.0, 1.0),
                2.6
            ) * edgeBand;
            float fresnel = mix(glassVolumeFresnel, legacyFresnel, solidMode);
            float solidSpecular = pow(facing, 3.5) * solidEdge;
            float solidFillSpecular = pow(oppositeFacing, 3.2) * solidEdge;
            float bevelShade = solidBevel * (1.0 - facing) *
                0.11 * solidMode;

            float3 colour = body;
            // Lift the transmissive face so wallpaper colour remains vivid
            // through the glyph instead of reading as a dark cut-out. This
            // multiplier is blended out completely for Solid/Frosted.
            float glassFaceLighting = 0.82 +
                keyDiffuse3D * 0.30 +
                fillDiffuse3D * 0.16;
            // Preserve the full diffuse lift in dark glass, where it prevents
            // the transmitted body from reading as a cut-out. Bright wallpaper
            // already carries enough energy: taper only the positive lift above
            // neutral there, so cyan/blue colour is less likely to clip white.
            // This is driven by the local transmitted colour, shared by all glyphs.
            float glassFaceInputLuma = dot(
                colour,
                float3(0.2126, 0.7152, 0.0722)
            );
            float glassFaceLightingHeadroom = 1.0 -
                smoothstep(0.46, 0.80, glassFaceInputLuma);
            float glassFaceLightingGain = max(glassFaceLighting - 1.0, 0.0);
            float glassFaceLightingPreserved =
                min(glassFaceLighting, 1.0) +
                glassFaceLightingGain * glassFaceLightingHeadroom;
            colour *= mix(glassFaceLightingPreserved, 1.0, solidMode);
            float directionalReflectionLuma = dot(
                directionalEnvironmentReflection,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 directionalReflectionChroma =
                directionalEnvironmentReflection -
                    float3(directionalReflectionLuma);
            float glassFaceLuma = dot(
                colour,
                float3(0.2126, 0.7152, 0.0722)
            );
            float glassFaceDarkness =
                1.0 - smoothstep(0.18, 0.52, glassFaceLuma);
            // Let optical thickness carry some reflected environment
            // across the broad front face, not only around the curved rim.
            // Curvature remains the primary driver; path scattering contributes
            // a restrained 55% coverage floor in thick glass. This keeps the
            // reflection tied to the same Beer-Lambert volume as transmission.
            float glassFaceReflectionCoverage = max(
                smoothstep(0.08, 0.48, glassProfileSlope),
                glassPathScatterRoughness * 0.55
            );
            float glassFaceReflectionAmount = mix(
                0.035,
                0.20,
                glassFaceReflectionCoverage
            ) * glassHorizontalHighlight;
            float glassChromaPickup = smoothstep(
                0.025,
                0.16,
                length(directionalReflectionChroma)
            );
            float glassReflectionLumaLift =
                max(directionalReflectionLuma - glassFaceLuma, 0.0) *
                glassFaceDarkness * glassFaceReflectionAmount * 0.18;
            colour = clamp(
                colour +
                    directionalReflectionChroma *
                        glassFaceReflectionAmount * glassChromaPickup * 1.34 +
                    float3(glassReflectionLumaLift),
                0.0,
                1.0
            );
            // Add a low-energy, directionally reflected wallpaper image across
            // the face. Unlike the view-ray sample near normal incidence, these
            // existing key/fill environment samples come from distinct offsets,
            // so they contribute a real spatial reflection pattern. A broad
            // microfacet lobe follows the same surface normal used by the glints.
            // Only positive reflected energy is returned, and dark-face gating
            // protects saturated/bright transmitted regions from whitening.
            float keyInteriorReflectionLobe = pow(
                max(dot(glassSurfaceNormal, keyHalfVector), 0.0),
                4.0
            );
            float fillInteriorReflectionLobe = pow(
                max(dot(glassSurfaceNormal, fillHalfVector), 0.0),
                4.0
            );
            float glassInteriorReflectionLobe = mix(
                fillInteriorReflectionLobe,
                keyInteriorReflectionLobe,
                0.64
            ) * glassHorizontalHighlight;
            // Add a second, broader microfacet lobe across front-facing glass.
            // The existing exponent-4 lobe preserves the more directional
            // reflection; this exponent-2 lobe fills in the broad face response
            // using the same measured key/fill environment and surface normal.
            // Couple its strength to Beer-Lambert path depth so it belongs to
            // the same optical volume, not a flat screen-space colour wash.
            float keyBroadFaceReflectionLobe = pow(
                max(dot(glassSurfaceNormal, keyHalfVector), 0.0),
                2.0
            );
            float fillBroadFaceReflectionLobe = pow(
                max(dot(glassSurfaceNormal, fillHalfVector), 0.0),
                2.0
            );
            float glassBroadFaceReflectionLobe = mix(
                fillBroadFaceReflectionLobe,
                keyBroadFaceReflectionLobe,
                0.64
            ) * glassHorizontalHighlight;
            float glassPathCoupledReflectionWeight =
                0.055 * glassBroadFaceReflectionLobe *
                glassPathScatterRoughness;
            // Keep the existing narrow and broad lobe coefficients, but let
            // each lobe select its own environment radiance. The shared scale
            // preserves the original combined 0.165 energy ceiling; applying
            // the face/material gate to both weights preserves the same dark-
            // face and Liquid-Glass-only constraints.
            float glassNarrowReflectionUnboundedWeight =
                0.11 * glassInteriorReflectionLobe;
            float glassBroadReflectionUnboundedWeight =
                glassPathCoupledReflectionWeight;
            float glassReflectionUnboundedWeight =
                glassNarrowReflectionUnboundedWeight +
                glassBroadReflectionUnboundedWeight;
            float glassReflectionEnergyScale = min(
                1.0,
                0.165 / max(glassReflectionUnboundedWeight, 0.0001)
            );
            float glassInteriorReflectionGate =
                glassFaceDarkness * (1.0 - solidMode);
            float glassNarrowReflectionWeight =
                glassNarrowReflectionUnboundedWeight *
                glassReflectionEnergyScale *
                glassInteriorReflectionGate;
            float glassBroadReflectionWeight =
                glassBroadReflectionUnboundedWeight *
                glassReflectionEnergyScale *
                glassInteriorReflectionGate;
            float glassInteriorReflectionWeight =
                glassNarrowReflectionWeight +
                glassBroadReflectionWeight;
            // Rough-surface reflection uses the anisotropic wallpaper
            // footprint already computed for transmission. It is the broad
            // lobe's source; the narrow lobe retains the directional/view
            // environment. Neither source adds a texture evaluation.
            float3 glassRoughReflectionEnvironment = mix(
                frostedWallpaper,
                microfacetTransmission,
                0.65
            );
            float3 glassNarrowReflectionSource = mix(
                directionalEnvironmentReflection,
                environmentReflection,
                0.34
            );
            // Preserve each lobe's radiance until after its nonlinear
            // positive-energy response. Averaging the two RGB sources first
            // can cancel a useful channel when one source is bright and the
            // other is dark, even though each lobe carries valid reflected
            // environment energy. Chroma remains a shared, linear transfer
            // before its one existing combined headroom clamp.
            float glassNarrowReflectionSourceLuma = dot(
                glassNarrowReflectionSource,
                float3(0.2126, 0.7152, 0.0722)
            );
            float glassBroadReflectionSourceLuma = dot(
                glassRoughReflectionEnvironment,
                float3(0.2126, 0.7152, 0.0722)
            );
            float glassReflectionFaceLuma = dot(
                colour,
                float3(0.2126, 0.7152, 0.0722)
            );
            float3 glassNarrowReflectionSourceChroma =
                glassNarrowReflectionSource -
                float3(glassNarrowReflectionSourceLuma);
            float3 glassBroadReflectionSourceChroma =
                glassRoughReflectionEnvironment -
                float3(glassBroadReflectionSourceLuma);
            float3 glassReflectionFaceChroma =
                colour - float3(glassReflectionFaceLuma);
            // Chroma transfer is linear in the source, so retain the exact
            // original combined transfer by summing the separately weighted
            // narrow and broad contributions before applying one shared clamp.
            float3 glassNarrowReflectedChromaTransfer =
                (glassNarrowReflectionSourceChroma -
                    glassReflectionFaceChroma) *
                glassNarrowReflectionWeight * 0.58;
            float3 glassBroadReflectedChromaTransfer =
                (glassBroadReflectionSourceChroma -
                    glassReflectionFaceChroma) *
                glassBroadReflectionWeight * 0.58;
            float3 glassReflectedChromaTransfer =
                glassNarrowReflectedChromaTransfer +
                glassBroadReflectedChromaTransfer;
            float3 glassChromaPositiveHeadroom =
                max(float3(1.0) - colour, float3(0.0)) * 0.16;
            float3 glassChromaNegativeHeadroom =
                max(colour, float3(0.0)) * 0.16;
            glassReflectedChromaTransfer = clamp(
                glassReflectedChromaTransfer,
                -glassChromaNegativeHeadroom,
                glassChromaPositiveHeadroom
            );
            // Positive reflected radiance is nonlinear: evaluate each
            // environment lobe against the unmodified face independently, then
            // sum the contributions under the original shared 14% headroom.
            float3 glassNarrowReflectionEnergy =
                max(
                    glassNarrowReflectionSource - colour,
                    float3(0.0)
                ) * glassNarrowReflectionWeight;
            float3 glassBroadReflectionEnergy =
                max(
                    glassRoughReflectionEnvironment - colour,
                    float3(0.0)
                ) * glassBroadReflectionWeight;
            float3 glassInteriorReflectionEnergy =
                glassNarrowReflectionEnergy +
                glassBroadReflectionEnergy;
            float3 glassInteriorReflectionHeadroom =
                max(float3(1.0) - colour, float3(0.0)) * 0.14;
            colour += min(
                glassInteriorReflectionEnergy,
                glassInteriorReflectionHeadroom
            );
            colour += glassReflectedChromaTransfer;
            colour -= float3(0.12, 0.15, 0.22) *
                glassSideShadow * (1.0 - solidMode) * 0.22;
            colour *= mix(0.985, 1.025, glassLens);
            colour += uAccentColor * fresnel * mix(0.68, 0.15, solidMode);

            // Preserve the separate Frosted Glass treatment. Liquid Glass
            // does not use an independent white bevel or dark perimeter.
            colour += float3(0.98, 0.99, 1.0) *
                solidBevel * 0.14 * solidMode;
            colour += float3(1.0, 1.0, 1.0) *
                solidSpecular * 0.62 * solidMode;
            colour += uAccentColor *
                solidFillSpecular * 0.15 * solidMode;

            // Fresnel adds the stronger grazing-angle reflection on top of
            // the subtler face reflection. The broad specular remains visible
            // on the front cap; tight glints stay confined to its shoulder.
            float finalFresnelAmount =
                glassVolumeFresnel * 0.15 * (1.0 - solidMode);
            colour = clamp(
                colour +
                    environmentReflectionChroma * finalFresnelAmount * 0.72,
                0.0,
                1.0
            );
            float glassRimLighting = glassRim *
                pow(max(dot(glassSurfaceNormal, keyLight3D), 0.0), 1.35) * glassHorizontalHighlight;
            // Let the broad shoulder reflection carry the colour of the
            // environment actually seen by the convex reflected ray, rather
            // than laying a fixed cool-white wash over the transmitted face.
            // Limit the additive return by per-channel headroom so it sits on
            // the coloured glass body without clipping it toward white.
            float glassAreaReflectionWeight =
                glassAreaReflection * (1.0 - solidMode) * 0.96;
            float3 glassAreaReflectionEnergy =
                chromaticEnvironmentReflection * glassAreaReflectionWeight;
            float3 glassAreaReflectionHeadroom =
                max(float3(1.0) - colour, float3(0.0));
            float3 glassAreaReflectionContribution = min(
                glassAreaReflectionEnergy,
                glassAreaReflectionHeadroom * 0.90
            );
            colour += glassAreaReflectionContribution;
            // Keep the narrow key glint crisp, but let its white core inherit
            // a restrained amount of the reflected environment's chroma.
            // This makes the highlight read as a reflection on coloured glass
            // instead of a fixed white streak laid over the transmitted face.
            float3 glassKeyGlintColour = clamp(
                float3(0.98, 0.99, 1.0) +
                    environmentReflectionChroma * 0.55,
                0.0,
                1.0
            );
            colour += glassKeyGlintColour *
                glassKeyGlint * (1.0 - solidMode) * 1.28;
            colour += float3(0.68, 0.82, 1.0) *
                glassFillGlint * (1.0 - solidMode) * 0.88;
            colour += float3(0.96, 0.98, 1.0) *
                glassRimLighting * (1.0 - solidMode) * 0.20;

            // Unified secondary bevel reflection: every glyph uses the same
            // height-map normal, curvature gate and light/view half-vector.
            // This naturally catches stroke ends and curved shoulders without
            // hard-coded digit identities or position-specific highlight axes.
            float glassBevelSpecular = pow(
                max(dot(glassSurfaceNormal, keyHalfVector), 0.0),
                18.0
            ) * smoothstep(0.08, 0.42, glassProfileSlope) *
                normalConfidence *
                mix(1.0, 0.72, horizontalShoulder);
            colour += environmentReflectionChroma *
                glassBevelSpecular * 0.42 * (1.0 - solidMode);
            // Keep the narrow bevel reflection, but let its soft highlight
            // follow the reflected environment instead of adding a fixed
            // pale-blue veil over the transmitted face. Per-channel headroom
            // protects the underlying colour while retaining a glass edge.
            float3 glassBevelReflectionColour = mix(
                float3(0.80, 0.91, 1.0),
                chromaticEnvironmentReflection,
                0.72
            );
            float3 glassBevelReflectionHeadroom =
                max(float3(1.0) - colour, float3(0.0)) * 0.70;
            float3 glassBevelReflectionContribution = min(
                glassBevelReflectionColour *
                    glassBevelSpecular * 0.12 * (1.0 - solidMode),
                glassBevelReflectionHeadroom
            );
            colour += glassBevelReflectionContribution;

            // Ambient glass scatter keeps digits legible over pure-black
            // wallpaper regions without laying a gray veil over bright images.
            // The vertical silver-blue gradient is strongest near the upper
            // half, and is confined to the inner face rather than its rim.
            float ambientClockY = p.y - uClockCenter.y * uResolution.y;
            float ambientVertical = clamp(
                0.5 + ambientClockY / max(uResolution.y * 0.17, 1.0),
                0.0,
                1.0
            );
            float3 ambientGradient = mix(
                float3(0.34, 0.37, 0.43),
                float3(0.21, 0.24, 0.30),
                ambientVertical
            );
            // Keep the dark-area lift chromatic instead of laying a neutral
            // gray veil over dark blue wallpaper visible through the glass.
            float3 ambientEnvironment = mix(
                ambientGradient,
                frostedWallpaper,
                0.72
            );
            float ambientFaceLuma = dot(
                colour,
                float3(0.2126, 0.7152, 0.0722)
            );
            float ambientDarkness = 1.0 -
                smoothstep(0.025, 0.20, ambientFaceLuma);
            float ambientInteriorMask = smoothstep(
                0.10,
                0.62,
                glassVolumeRegion
            );
            float ambientScatter = 0.42 *
                ambientDarkness *
                ambientInteriorMask *
                (1.0 - solidMode);
            colour = mix(
                colour,
                ambientEnvironment,
                ambientScatter
            );

            colour -= float3(0.13, 0.15, 0.21) * bevelShade;
            // These perimeter strokes belong only to Frosted Glass; applying
            // them to Liquid Glass made it look like embossed text.
            colour -= float3(0.08, 0.09, 0.14) *
                edgeBand * (1.0 - facing) * 0.055 * solidMode;
            colour += float3(1.0, 1.0, 1.0) *
                outerLip * 0.07 * solidMode;

            // Keep a restrained rim only for the Frosted Glass option.
            float solidRim = exp(-abs(signedDistance - 2.5) / 4.6) *
                pow(max(dot(normal, keyLight), 0.0), 1.45);
            colour += float3(1.0, 1.0, 1.0) *
                solidRim * 0.25 * solidMode;

            // Frosted Glass keeps its separate, subdued satin highlight.
            // This coordinate ribbon is retained only for that material.
            float2 clockRelative = p - uClockCenter * uResolution;
            float reflectionAxis = clockRelative.x * -0.42 +
                clockRelative.y * 0.91 + sin(uTime * 0.16) * 18.0;
            float solidSheen = exp(
                -pow((reflectionAxis - 18.0) / 66.0, 2.0)
            ) * glassLens;
            colour += float3(1.0, 1.0, 1.0) *
                solidSheen * solidMode * 0.16;

            // The shader replaces the backdrop pixels inside the glyph with
            // their refracted samples. Full mask alpha is required; the glass
            // remains optically clear because its RGB is the warped wallpaper,
            // not a white translucent paint layer.
            float glassOpacity = 1.0;
            float solidOpacity = 0.78;
            float bodyOpacity = mix(glassOpacity, solidOpacity, solidMode);
            float alpha = clamp(
                coverage * (
                    bodyOpacity +
                    edgeBand * mix(0.31, 0.10, solidMode) +
                    outerLip * mix(0.035, 0.07, solidMode)
                ),
                0.0,
                1.0
            );

            // RuntimeShader outputs use premultiplied alpha.
            float3 premultipliedColour = clamp(colour, 0.0, 1.0) * alpha;
            return half4(premultipliedColour, alpha);
        }
    """.trimIndent()
    }

    private var surfaceWidth = 0
    private var surfaceHeight = 0
    private var backdrop: Bitmap? = null
    private var wallpaperSample: Bitmap? = null
    private var foregroundOcclusionMask: Bitmap? = null
    private var glyphMask: Bitmap? = null
    private var colonMask: Bitmap? = null
    private var colonMaskOriginX = 0f
    private var colonMaskOriginY = 0f
    private var glyphSdf: Bitmap? = null
    private val retiredGlyphMasks = mutableListOf<Bitmap>()
    private val retiredColonMasks = mutableListOf<Bitmap>()
    private val retiredGlyphSdfs = mutableListOf<Bitmap>()
    private var glyphSdfOriginX = 0f
    private var glyphSdfOriginY = 0f
    private var glyphSdfScale = 1f
    private var glyphSdfStale = false
    private var deviceTiltX = 0f
    private var deviceTiltY = 0f
    private var adaptiveAccentColor: Int = Color.rgb(222, 130, 177)
    private var clockFontStyle = 0
    private var clockFontWeight = 640
    private var clockColor = Color.WHITE
    private var glassMode = true
    private var glassShader: RuntimeShader? = null
    private var shaderFailureReported = false
    private var cachedTimeText: String? = null
    private var clockCenterX = DEFAULT_CLOCK_X
    private var clockCenterY = DEFAULT_CLOCK_Y
    private var clockScale = 1f
    private var maskClockScale = 1f
    private var pinchStartClockScale = 1f
    private var clockWidthScale = 1f
    private var maskClockWidthScale = 1f
    private var pinchStartClockWidthScale = 1f
    private var gestureOffsetX = 0f
    private var gestureOffsetY = 0f
    private var previewScaleX = 1f
    private var previewScaleY = 1f

    private val glyphBaseBounds = RectF()
    private val glyphBounds = RectF()
    private val glyphBoundsSource = Rect()
    private val fallbackTypeface: Typeface = loadFallbackTypeface()
    private val typefaceCache = LinkedHashMap<Int, Typeface>()

    private val bitmapPaint = Paint(
        Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
    )

    private val shaderPaint = Paint(
        Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
    )

    private val shadowPaint = Paint(
        Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
    ).apply {
        colorFilter = PorterDuffColorFilter(
            Color.argb(38, 3, 9, 22),
            PorterDuff.Mode.SRC_IN
        )
    }

    private val clockOcclusionPaint = Paint(
        Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
    ).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_OUT)
    }

    private val fallbackGlyphPaint = Paint(
        Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
    ).apply {
        color = Color.WHITE
        alpha = 218
    }

    private val clockLayerBounds = RectF()

    private val clockPreferences = context.getSharedPreferences(
        wallpaperContent.compositionPreferencesName,
        Context.MODE_PRIVATE
    )

    init {
        clockCenterX = clockPreferences.getFloat("center_x", DEFAULT_CLOCK_X)
            .coerceIn(0.10f, 0.90f)
        val savedCenterY = clockPreferences.getFloat("center_y", LEGACY_DEFAULT_CLOCK_Y)
        val positionMigrationDone = clockPreferences.getBoolean("reference_position_v1", false)
        val studioPositionMigrationDone =
            clockPreferences.getBoolean("clock_studio_safe_position_v2", false)
        // Reposition only the old untouched defaults so the entire live clock
        // remains visible above the appearance sheet. Preserve custom positions.
        clockCenterY = if (
            !studioPositionMigrationDone &&
            (savedCenterY == PREVIOUS_DEFAULT_CLOCK_Y ||
                (!positionMigrationDone && savedCenterY == LEGACY_DEFAULT_CLOCK_Y))
        ) {
            DEFAULT_CLOCK_Y
        } else {
            savedCenterY.coerceIn(0.10f, 0.78f)
        }
        clockScale = clockPreferences.getFloat("clock_scale", 1f)
            .coerceIn(MIN_CLOCK_SCALE, MAX_CLOCK_SCALE)
        clockWidthScale = clockPreferences.getFloat("clock_width_scale", 1f)
            .coerceIn(MIN_CLOCK_WIDTH_SCALE, MAX_CLOCK_WIDTH_SCALE)
        readAppearancePreferences()
        if (!positionMigrationDone || !studioPositionMigrationDone) {
            clockPreferences.edit()
                .putBoolean("reference_position_v1", true)
                .putBoolean("clock_studio_safe_position_v2", true)
                .putFloat("center_y", clockCenterY)
                .apply()
        }
    }

    fun setViewport(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        if (width == surfaceWidth && height == surfaceHeight && backdrop != null) return

        surfaceWidth = width
        surfaceHeight = height
        rebuildBackdrop()
    }

    fun reloadWallpaper() {
        rebuildBackdrop()
    }

    fun reloadAppliedState() {
        clockCenterX = clockPreferences.getFloat("center_x", DEFAULT_CLOCK_X)
            .coerceIn(0.10f, 0.90f)
        clockCenterY = clockPreferences.getFloat("center_y", DEFAULT_CLOCK_Y)
            .coerceIn(0.10f, 0.78f)
        clockScale = clockPreferences.getFloat("clock_scale", 1f)
            .coerceIn(MIN_CLOCK_SCALE, MAX_CLOCK_SCALE)
        clockWidthScale = clockPreferences.getFloat("clock_width_scale", 1f)
            .coerceIn(MIN_CLOCK_WIDTH_SCALE, MAX_CLOCK_WIDTH_SCALE)
        readAppearancePreferences()
        rebuildBackdrop()
    }

    /**
     * Apply the Font & Color sheet without restarting the wallpaper service.
     * Typeface and colour settings are stored alongside the clock composition.
     */
    fun reloadAppearance(rebuildDistanceField: Boolean = true) {
        diagnostics.record("Renderer reload start: sdf=$rebuildDistanceField surface=${surfaceWidth}x${surfaceHeight}")
        val previousStyle = clockFontStyle
        val previousWeight = clockFontWeight
        val previousColor = clockColor
        val previousGlassMode = glassMode

        readAppearancePreferences()
        val glyphAppearanceChanged =
            previousStyle != clockFontStyle || previousWeight != clockFontWeight

        if (glyphAppearanceChanged) {
            typefaceCache.clear()
            cachedTimeText = null
            if (surfaceWidth > 0 && surfaceHeight > 0) {
                try {
                    rebuildGlyphMask(
                        force = true,
                        rebuildDistanceField = rebuildDistanceField
                    )
                } catch (memoryError: OutOfMemoryError) {
                    diagnostics.report("renderer-glyph-mask-out-of-memory", memoryError)
                    Log.e(TAG, "Clock weight preview ran out of memory; keeping the last rendered mask.", memoryError)
                } catch (exception: RuntimeException) {
                    diagnostics.report("renderer-glyph-mask", exception)
                    Log.e(TAG, "Clock weight preview could not rebuild its glyph mask.", exception)
                }
            }
        } else if (rebuildDistanceField && glyphSdfStale && surfaceWidth > 0) {
            rebuildGlyphSdf()
        }

        if (glassShader == null && surfaceWidth > 0 && surfaceHeight > 0) {
            rebuildGlassShader()
        } else if (
            glyphAppearanceChanged ||
            previousColor != clockColor ||
            previousGlassMode != glassMode
        ) {
            updateGlassShaderAppearance()
        }
    }

    private fun updateGlassShaderAppearance() {
        val shader = glassShader ?: return
        try {
            shader.setFloatUniform(
                "uAccentColor",
                Color.red(adaptiveAccentColor) / 255f,
                Color.green(adaptiveAccentColor) / 255f,
                Color.blue(adaptiveAccentColor) / 255f
            )
            shader.setFloatUniform(
                "uClockTint",
                Color.red(clockColor) / 255f,
                Color.green(clockColor) / 255f,
                Color.blue(clockColor) / 255f
            )
            shader.setFloatUniform(
                "uTintStrength",
                if (clockColor == Color.WHITE) 0f else 0.16f
            )
            shader.setFloatUniform("uFrosted", if (glassMode) 0f else 1f)
        } catch (exception: RuntimeException) {
            Log.w(TAG, "Could not update clock material uniforms.", exception)
            rebuildGlassShader()
        }
    }

    private fun readAppearancePreferences() {
        clockFontStyle = clockPreferences.getInt("clock_font_style", 0).coerceIn(0, 3)
        clockFontWeight = clockPreferences.getInt("clock_font_weight", 400)
            .coerceIn(100, 900)
        clockColor = clockPreferences.getInt("clock_color", Color.WHITE)
        glassMode = clockPreferences.getBoolean("clock_glass_mode", true)
    }

    fun isClockHit(x: Float, y: Float): Boolean {
        if (glyphBounds.isEmpty) return false
        val hit = RectF(glyphBounds)
        val margin = (context.resources.displayMetrics.density * 44f)
        hit.inset(-margin, -margin)
        return hit.contains(x, y)
    }

    fun moveClockBy(deltaX: Float, deltaY: Float) {
        if (surfaceWidth <= 0 || surfaceHeight <= 0) return

        val previousX = clockCenterX
        val previousY = clockCenterY
        clockCenterX = (clockCenterX + deltaX / surfaceWidth)
            .coerceIn(0.10f, 0.90f)
        clockCenterY = (clockCenterY + deltaY / surfaceHeight)
            .coerceIn(0.10f, 0.78f)

        // Keep the cached glyph/SDF textures. Moving a canvas transform is
        // effectively free compared with rebuilding two full distance fields.
        gestureOffsetX += (clockCenterX - previousX) * surfaceWidth
        gestureOffsetY += (clockCenterY - previousY) * surfaceHeight
        updatePreviewBounds()
    }

    fun beginScaleGesture() {
        pinchStartClockScale = clockScale
    }

    fun scaleClock(distanceRatio: Float) {
        if (!distanceRatio.isFinite() || distanceRatio <= 0f) return
        // Use the original scale at pointer-down, not accumulated MOVE deltas.
        // That prevents event-rate jitter and lets every new pinch start cleanly.
        val requestedScale = (pinchStartClockScale * distanceRatio)
            .coerceIn(MIN_CLOCK_SCALE, MAX_CLOCK_SCALE)
        // Derive the pinch ceiling from the actual rendered glyph bounds,
        // not a guessed font-size fraction. This keeps the preview and final
        // committed mask on exactly the same vertical limit.
        val maxPreviewRatio = if (!glyphBaseBounds.isEmpty && surfaceHeight > 0) {
            (surfaceHeight * MAX_COMMITTED_HEIGHT_FRACTION /
                glyphBaseBounds.height().coerceAtLeast(1f)).coerceAtLeast(1f)
        } else {
            MAX_CLOCK_SCALE / maskClockScale.coerceAtLeast(0.01f)
        }
        previewScaleY = (requestedScale / maskClockScale.coerceAtLeast(0.01f))
            .coerceIn(0.25f, maxPreviewRatio)
        clockScale = (maskClockScale * previewScaleY)
            .coerceIn(MIN_CLOCK_SCALE, MAX_CLOCK_SCALE)
        updatePreviewBounds()
    }

    fun beginWidthScaleGesture() {
        pinchStartClockWidthScale = clockWidthScale
    }

    fun scaleClockWidth(distanceRatio: Float) {
        if (!distanceRatio.isFinite() || distanceRatio <= 0f) return

        val maxAllowedScale = if (!glyphBaseBounds.isEmpty && surfaceWidth > 0) {
            (
                surfaceWidth * MAX_COMMITTED_WIDTH_FRACTION *
                    maskClockWidthScale /
                    glyphBaseBounds.width().coerceAtLeast(1f)
                ).coerceIn(MIN_CLOCK_WIDTH_SCALE, MAX_CLOCK_WIDTH_SCALE)
        } else {
            MAX_CLOCK_WIDTH_SCALE
        }

        clockWidthScale = (pinchStartClockWidthScale * distanceRatio)
            .coerceIn(MIN_CLOCK_WIDTH_SCALE, maxAllowedScale)
        previewScaleX = clockWidthScale / maskClockWidthScale.coerceAtLeast(0.01f)
        updatePreviewBounds()
    }

    fun finishGesture() {
        if (surfaceWidth <= 0 || surfaceHeight <= 0) return
        rebuildGlyphMask(force = true)
    }

    private fun updatePreviewBounds() {
        if (glyphBaseBounds.isEmpty) {
            glyphBounds.setEmpty()
            return
        }
        glyphBounds.set(glyphBaseBounds)
        val transform = Matrix().apply {
            setScale(
                previewScaleX,
                previewScaleY,
                glyphBaseBounds.centerX(),
                glyphBaseBounds.centerY()
            )
        }
        transform.mapRect(glyphBounds)
        glyphBounds.offset(gestureOffsetX, gestureOffsetY)
    }

    fun setDeviceTilt(x: Float, y: Float) {
        if (!x.isFinite() || !y.isFinite()) return
        deviceTiltX += (x.coerceIn(-1f, 1f) - deviceTiltX) * 0.22f
        deviceTiltY += (y.coerceIn(-1f, 1f) - deviceTiltY) * 0.22f
    }

    fun persistComposition() {
        clockPreferences.edit()
            .putFloat("center_x", clockCenterX)
            .putFloat("center_y", clockCenterY)
            .putFloat("clock_scale", clockScale)
            .putFloat("clock_width_scale", clockWidthScale)
            .apply()
    }

    fun draw(canvas: Canvas, timeSeconds: Float) {
        val width = surfaceWidth
        val height = surfaceHeight
        if (width <= 0 || height <= 0) return

        rebuildGlyphMask(force = false)

        // Keep the original selected wallpaper as the only background image.
        // The user selection is an alpha mask applied to the clock layer only.
        val background = backdrop ?: return
        canvas.drawBitmap(background, 0f, 0f, bitmapPaint)

        val mask = glyphMask ?: return
        if (glyphBounds.isEmpty) return

        val shadowOffset = if (glassMode) {
            max(4f, height * 0.0022f)
        } else {
            max(2f, height * 0.0015f)
        }
        shadowPaint.colorFilter = PorterDuffColorFilter(
            Color.argb(if (glassMode) 58 else 38, 3, 9, 22),
            PorterDuff.Mode.SRC_IN
        )
        clockLayerBounds.set(glyphBounds)
        clockLayerBounds.inset(-shadowOffset * 2f, -shadowOffset * 2f)
        clockLayerBounds.intersect(
            0f,
            0f,
            width.toFloat(),
            height.toFloat()
        )

        // Draw the clock in isolation. DST_OUT then removes only the clock
        // pixels beneath the user-selected object. The untouched wallpaper
        // remains visible below, so the selected object never replaces it.
        val layerSave = canvas.saveLayer(clockLayerBounds, null)
        try {
            val transformSave = canvas.save()
            canvas.translate(gestureOffsetX, gestureOffsetY)
            canvas.scale(
                previewScaleX,
                previewScaleY,
                glyphBaseBounds.centerX(),
                glyphBaseBounds.centerY()
            )

            val shadowSave = canvas.save()
            canvas.translate(0f, shadowOffset)
            canvas.drawBitmap(mask, 0f, 0f, shadowPaint)
            canvas.restoreToCount(shadowSave)

            var shaderDrawn = false
            val shader = glassShader
            if (canvas.isHardwareAccelerated && shader != null) {
                try {
                    shader.setFloatUniform(
                        "uResolution",
                        width.toFloat(),
                        height.toFloat()
                    )
                    shader.setFloatUniform("uTime", timeSeconds)
                    shader.setFloatUniform("uTilt", deviceTiltX, deviceTiltY)
                    shader.setFloatUniform(
                        "uClockCenter",
                        glyphBaseBounds.centerX() / width.toFloat(),
                        glyphBaseBounds.centerY() / height.toFloat()
                    )
                    shaderPaint.shader = shader
                    canvas.drawRect(glyphBaseBounds, shaderPaint)
                    shaderPaint.shader = null
                    shaderDrawn = true
                } catch (exception: RuntimeException) {
                    shaderPaint.shader = null
                    if (!shaderFailureReported) {
                        shaderFailureReported = true
                        diagnostics.report("agsl-render-frame", exception)
                    }
                    Log.w(TAG, "AGSL rendering failed; showing a visible clock fallback.", exception)
                }
            }

            // During a pinch/drag, transform the cached mask and shader rather
            // than regenerating their pixel data on the UI thread.
            if (!shaderDrawn) {
                fallbackGlyphPaint.color = clockColor
                fallbackGlyphPaint.alpha = if (glassMode) 218 else 232
                canvas.drawBitmap(mask, 0f, 0f, fallbackGlyphPaint)
            }
            canvas.restoreToCount(transformSave)

            // Keep the selected foreground object fixed in wallpaper space;
            // only the clock layer is transformed beneath its depth mask.
            foregroundOcclusionMask?.let { occlusionMask ->
                canvas.drawBitmap(occlusionMask, 0f, 0f, clockOcclusionPaint)
            }
        } finally {
            canvas.restoreToCount(layerSave)
            shaderPaint.shader = null
        }
    }

    fun release() {
        glassShader = null
        glyphMask?.takeUnless { it.isRecycled }?.recycle()
        glyphMask = null
        colonMask?.takeUnless { it.isRecycled }?.recycle()
        colonMask = null
        retiredColonMasks.forEach { if (!it.isRecycled) it.recycle() }
        retiredColonMasks.clear()
        glyphSdf?.takeUnless { it.isRecycled }?.recycle()
        glyphSdf = null
        retiredGlyphMasks.forEach { if (!it.isRecycled) it.recycle() }
        retiredGlyphMasks.clear()
        retiredGlyphSdfs.forEach { if (!it.isRecycled) it.recycle() }
        retiredGlyphSdfs.clear()
        backdrop?.recycle()
        if (wallpaperSample !== backdrop) wallpaperSample?.recycle()
        foregroundOcclusionMask?.recycle()
        backdrop = null
        wallpaperSample = null
        foregroundOcclusionMask = null
        cachedTimeText = null
    }

    private fun rebuildBackdrop() {
        val width = surfaceWidth
        val height = surfaceHeight
        if (width <= 0 || height <= 0) return

        val wallpaperFile = context.filesDir.resolve(wallpaperContent.sourceFileName)
        val manualMaskFile = context.filesDir.resolve(wallpaperContent.manualForegroundFileName)

        val decodedWallpaper = if (
            wallpaperFile.exists() && wallpaperFile.length() > 0L
        ) {
            decodeWallpaper(wallpaperFile.absolutePath, width, height)
        } else {
            null
        }

        val decodedMask = manualMaskFile
            .takeIf { it.exists() && it.length() > 0L }
            ?.let { decodeWallpaper(it.absolutePath, width, height) }

        val wallpaperOutput = Bitmap.createBitmap(
            width,
            height,
            Bitmap.Config.ARGB_8888
        )

        if (decodedWallpaper != null) {
            drawCoverBitmap(Canvas(wallpaperOutput), decodedWallpaper, width, height)
            decodedWallpaper.recycle()
        } else {
            Log.e(TAG, "Selected wallpaper could not be decoded; using the fallback background.")
            drawDefaultBackdrop(Canvas(wallpaperOutput), width, height)
        }

        // This is only a per-pixel alpha mask. Never draw it as an image.
        // Its alpha erases clock pixels where the selected object overlaps.
        val occlusionOutput = decodedMask?.let { sourceMask ->
            Bitmap.createBitmap(
                width,
                height,
                Bitmap.Config.ARGB_8888
            ).also { output ->
                drawCoverBitmap(Canvas(output), sourceMask, width, height)
                sourceMask.recycle()
            }
        }

        backdrop?.recycle()
        if (wallpaperSample !== backdrop) wallpaperSample?.recycle()
        foregroundOcclusionMask?.recycle()

        adaptiveAccentColor = extractWallpaperAccent(wallpaperOutput)
        backdrop = wallpaperOutput
        wallpaperSample = wallpaperOutput
        foregroundOcclusionMask = occlusionOutput

        cachedTimeText = null
        rebuildGlyphMask(force = true)
        rebuildGlassShader()
    }

    private fun drawCoverBitmap(
        canvas: Canvas,
        bitmap: Bitmap,
        width: Int,
        height: Int
    ) {
        val scale = max(
            width.toFloat() / bitmap.width,
            height.toFloat() / bitmap.height
        )
        val dx = (width - bitmap.width * scale) / 2f
        val dy = (height - bitmap.height * scale) / 2f
        val matrix = Matrix().apply {
            setScale(scale, scale)
            postTranslate(dx, dy)
        }
        canvas.drawBitmap(bitmap, matrix, bitmapPaint)
    }

    private fun decodeWallpaper(
        path: String,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeFile(path, bounds)
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

            var sampleSize = 1
            while (
                bounds.outWidth / (sampleSize * 2) >= targetWidth * 2 ||
                bounds.outHeight / (sampleSize * 2) >= targetHeight * 2
            ) {
                sampleSize *= 2
            }

            BitmapFactory.decodeFile(
                path,
                BitmapFactory.Options().apply {
                    inSampleSize = sampleSize
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inDither = true
                }
            )
        } catch (exception: Exception) {
            Log.e(TAG, "Could not decode wallpaper image.", exception)
            null
        }
    }

    private fun drawDefaultBackdrop(
        canvas: Canvas,
        width: Int,
        height: Int
    ) {
        val base = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = LinearGradient(
                0f,
                0f,
                width.toFloat(),
                height.toFloat(),
                intArrayOf(
                    Color.rgb(3, 8, 20),
                    Color.rgb(14, 18, 48),
                    Color.rgb(47, 8, 67),
                    Color.rgb(6, 5, 18)
                ),
                null,
                Shader.TileMode.CLAMP
            )
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), base)

        val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            shader = RadialGradient(
                width * 0.78f,
                height * 0.43f,
                width * 0.78f,
                intArrayOf(
                    Color.argb(105, 72, 102, 181),
                    Color.argb(0, 72, 102, 181)
                ),
                null,
                Shader.TileMode.CLAMP
            )
        }
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), glow)
    }

    private fun rebuildGlyphMask(
        force: Boolean,
        rebuildDistanceField: Boolean = true
    ) {
        val timeText = currentTimeText()
        if (!force && timeText == cachedTimeText && glyphMask != null) return

        val width = surfaceWidth
        val height = surfaceHeight
        if (width <= 0 || height <= 0) return

        val previousMask = glyphMask
        val previousColonMask = colonMask

        val mask = try {
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        } catch (memoryError: OutOfMemoryError) {
            Log.e(TAG, "Unable to allocate the clock glyph mask at ${width}x${height}.", memoryError)
            return
        }
        val maskCanvas = Canvas(mask)
        maskCanvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val glyphs = timeText.map { it.toString() }
        // The expansion parameter changes the optical width of each numeral
        // independently. The clock becomes taller as the user enlarges it,
        // while the digits gain carefully controlled width rather than being
        // uniformly stretched into a wider line.
        val expansion = ((clockScale - 1f) / (MAX_CLOCK_SCALE - 1f))
            .coerceIn(0f, 1f)
        // The official Apple SF Pro variable font supplies real wght and wdth
        // axes. User-selected weight and width are applied before the adaptive
        // per-glyph optical shaping below.
        val fontWeight = clockFontWeight
        val fontWidth = clockFontWidthCoordinate()
        val maxWidth = width * MAX_GLYPH_WIDTH_FRACTION
        val maxBaseHeight = min(height * MAX_GLYPH_HEIGHT_FRACTION, width * 0.90f)
        // Kerning is based on actual visible outline bounds, not the font's
        // nominal advances. This removes built-in side-bearing gaps directly.
        val hourPairGapEm = 0.053f
        val minutePairGapEm = 0.019f
        val hourToColonGapEm = 0.038f
        val colonToMinuteGapEm = 0.034f
        val colonRadiusEm = COLON_DOT_RADIUS_EM * 1.08f

        val sizingPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.SUBPIXEL_TEXT_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.LEFT
            style = Paint.Style.FILL
            typeface = typefaceFor(fontWeight, 28, fontWidth)
            isSubpixelText = true
        }

        fun visibleGlyphBounds(size: Float): List<Pair<Float, Float>> {
            sizingPaint.textSize = size
            return glyphs.map { glyph ->
                if (glyph == ":") {
                    val radius = size * colonRadiusEm
                    -radius to radius
                } else {
                    val path = Path()
                    sizingPaint.getTextPath(glyph, 0, glyph.length, 0f, 0f, path)
                    val bounds = RectF()
                    path.computeBounds(bounds, true)
                    if (bounds.isEmpty) {
                        0f to (sizingPaint.measureText(glyph) *
                            glyphWidthProfile(glyph, expansion))
                    } else {
                        val compactWidth = glyphWidthProfile(glyph, expansion)
                        val center = bounds.centerX()
                        (center + (bounds.left - center) * compactWidth) to
                            (center + (bounds.right - center) * compactWidth)
                    }
                }
            }
        }

        fun visualGlyphOrigins(
            bounds: List<Pair<Float, Float>>,
            size: Float
        ): FloatArray {
            val origins = FloatArray(glyphs.size)
            if (glyphs.isEmpty()) return origins
            origins[0] = -bounds[0].first
            for (index in 1 until glyphs.size) {
                val previousGlyph = glyphs[index - 1]
                val currentGlyph = glyphs[index]
                val gapEm = when {
                    previousGlyph == ":" -> colonToMinuteGapEm
                    currentGlyph == ":" -> hourToColonGapEm
                    glyphs.take(index).contains(":") -> minutePairGapEm
                    else -> hourPairGapEm
                }
                val gap = size * gapEm
                val previousRight = origins[index - 1] + bounds[index - 1].second
                origins[index] = previousRight + gap - bounds[index].first
            }
            return origins
        }

        fun measureLine(size: Float): Float {
            val bounds = visibleGlyphBounds(size)
            val origins = visualGlyphOrigins(bounds, size)
            val visualWidth = if (glyphs.isEmpty()) 0f
                else origins.last() + bounds.last().second
            return visualWidth * CLOCK_GLYPH_WIDTH_SCALE
        }

        // Fit the expanded per-digit advances to the usable line width. This
        // keeps the clock centred and prevents minute digits colliding as the
        // optical profiles transition from compact to expanded.
        var low = 1f
        var high = min(maxBaseHeight, width * 0.90f)
        repeat(28) {
            val candidate = (low + high) * 0.5f
            if (measureLine(candidate) <= maxWidth) low = candidate else high = candidate
        }
        val fontSize = low
        sizingPaint.textSize = fontSize

        val metrics = sizingPaint.fontMetrics
        val centerX = clockCenterX * width
        val centerY = clockCenterY * height
        val baseline = centerY - (metrics.ascent + metrics.descent) * 0.5f
        val finalVisibleBounds = visibleGlyphBounds(fontSize)
        val relativeOrigins = visualGlyphOrigins(finalVisibleBounds, fontSize)
        val totalWidth = if (glyphs.isEmpty()) 0f
            else relativeOrigins.last() + finalVisibleBounds.last().second
        val lineOffsetX = centerX - totalWidth * 0.5f
        val glyphOrigins = relativeOrigins.map { it + lineOffsetX }

        val composedPath = Path().apply { fillType = Path.FillType.WINDING }
        val digitTwoPaths = mutableListOf<Path>()
        val composedBounds = RectF()
        val colonCenters = mutableListOf<Pair<Float, Float>>()
        var hasBounds = false

        glyphs.forEachIndexed { index, glyph ->
            if (glyph == ":") {
                val dotOffset = fontSize * COLON_DOT_OFFSET_EM
                val dotX = glyphOrigins[index]
                colonCenters += dotX to (centerY - dotOffset)
                colonCenters += dotX to (centerY + dotOffset)
            } else {
                // Generate the character's real SF Pro outline once. A
                // progressively condensed horizontal transform adapts the
                // *shape* of each digit as vertical expansion increases;
                // the vertical expansion below then makes the tall iOS form.
                val glyphPaint = Paint(sizingPaint).apply {
                    typeface = typefaceFor(fontWeight, 28, fontWidth)
                    textSize = fontSize
                    style = Paint.Style.FILL
                    strokeWidth = 0f
                }
                val glyphPath = Path()
                glyphPaint.getTextPath(
                    glyph,
                    0,
                    glyph.length,
                    glyphOrigins[index],
                    baseline,
                    glyphPath
                )

                val pathBounds = RectF()
                glyphPath.computeBounds(pathBounds, true)
                if (!pathBounds.isEmpty) {
                    // Every numeral, including "1", uses the actual outline from
                    // Apple's official SF Pro variable font. Optical transforms
                    // adjust layout only; they do not replace glyph artwork.
                    val compactWidth = glyphWidthProfile(glyph, expansion)
                    val opticalPath = Path()
                    val shapeTransform = Matrix().apply {
                        setScale(
                            compactWidth,
                            glyphHeightFactor(glyph),
                            pathBounds.centerX(),
                            pathBounds.centerY()
                        )
                    }
                    glyphPath.transform(shapeTransform, opticalPath)
                    if (glyph == "2") {
                        digitTwoPaths += Path(opticalPath)
                    } else {
                        composedPath.addPath(opticalPath)
                    }

                    val opticalBounds = RectF()
                    opticalPath.computeBounds(opticalBounds, true)
                    if (!opticalBounds.isEmpty) {
                        if (!hasBounds) {
                            composedBounds.set(opticalBounds)
                            hasBounds = true
                        } else {
                            composedBounds.union(opticalBounds)
                        }
                    }
                }
            }
        }

        if (hasBounds) {
            // Horizontal pinch is independent from the vertical clock scale.
            // Clamp against the actual glyph outline after viewport changes.
            val maxSafeWidthScale = (
                width * MAX_COMMITTED_WIDTH_FRACTION /
                    (composedBounds.width().coerceAtLeast(1f) *
                        CLOCK_GLYPH_WIDTH_SCALE)
                ).coerceIn(MIN_CLOCK_WIDTH_SCALE, MAX_CLOCK_WIDTH_SCALE)
            clockWidthScale = clockWidthScale.coerceAtMost(maxSafeWidthScale)

            // Apply native optical proportions and independent X/Y scales.
            val naturalShapeTransform = Matrix().apply {
                setScale(
                    CLOCK_GLYPH_WIDTH_SCALE * clockWidthScale,
                    CLOCK_GLYPH_HEIGHT_SCALE,
                    centerX,
                    centerY
                )
            }
            val naturalShapePath = Path()
            composedPath.transform(naturalShapeTransform, naturalShapePath)
            naturalShapePath.computeBounds(composedBounds, true)

            val naturalDigitTwoPaths = digitTwoPaths.map { sourcePath ->
                Path().also { sourcePath.transform(naturalShapeTransform, it) }
            }
            naturalDigitTwoPaths.forEach { twoPath ->
                val twoBounds = RectF()
                twoPath.computeBounds(twoBounds, true)
                composedBounds.union(twoBounds)
            }

            val dotRadius = fontSize * COLON_DOT_RADIUS_EM * 1.08f
            val naturalDotOffset = fontSize * COLON_DOT_OFFSET_EM * CLOCK_GLYPH_HEIGHT_SCALE
            colonCenters.forEach { (dotX, originalY) ->
                val naturalX = centerX +
                    (dotX - centerX) * CLOCK_GLYPH_WIDTH_SCALE * clockWidthScale
                val naturalY = centerY +
                    (originalY - centerY) * CLOCK_GLYPH_HEIGHT_SCALE
                val bounds = RectF(
                    naturalX - dotRadius,
                    naturalY - dotRadius,
                    naturalX + dotRadius,
                    naturalY + dotRadius
                )
                composedBounds.union(bounds)
            }

            // Enforce the safe viewport limit against the actual expanded
            // outlines, including the colon, before committing the scale.
            val allowedScaleY = (
                height * MAX_COMMITTED_HEIGHT_FRACTION /
                    composedBounds.height().coerceAtLeast(1f)
                ).coerceAtLeast(1f)
            val committedScaleY = clockScale.coerceAtMost(allowedScaleY)
            if (committedScaleY != clockScale) {
                clockScale = committedScaleY.coerceIn(MIN_CLOCK_SCALE, MAX_CLOCK_SCALE)
            }

            val verticalTransform = Matrix().apply {
                setScale(1f, committedScaleY, centerX, centerY)
            }
            val finalPath = Path()
            naturalShapePath.transform(verticalTransform, finalPath)
            val finalDigitTwoPaths = naturalDigitTwoPaths.map { sourcePath ->
                Path().also { sourcePath.transform(verticalTransform, it) }
            }
            val finalColonCenters = mutableListOf<Pair<Float, Float>>()

            // Keep the two colon dots circular as the clock grows. Only their
            // separation follows the vertical expansion.
            colonCenters.forEach { (dotX, originalY) ->
                val naturalX = centerX +
                    (dotX - centerX) * CLOCK_GLYPH_WIDTH_SCALE * clockWidthScale
                val naturalY = centerY + (originalY - centerY) * CLOCK_GLYPH_HEIGHT_SCALE
                val finalY = centerY + (naturalY - centerY) * committedScaleY
                finalPath.addCircle(naturalX, finalY, dotRadius, Path.Direction.CW)
                finalColonCenters += naturalX to finalY
            }
            val colonMaskData = createColonMask(
                finalColonCenters,
                dotRadius,
                width,
                height
            )
            colonMask = colonMaskData.first
            colonMaskOriginX = colonMaskData.second
            colonMaskOriginY = colonMaskData.third

            val finalBounds = RectF()
            finalPath.computeBounds(finalBounds, true)
            finalDigitTwoPaths.forEach { twoPath ->
                val twoBounds = RectF()
                twoPath.computeBounds(twoBounds, true)
                finalBounds.union(twoBounds)
            }
            val inset = max(2f, fontSize * 0.025f)
            glyphBounds.set(
                (finalBounds.left - inset).coerceAtLeast(0f),
                (finalBounds.top - inset).coerceAtLeast(0f),
                (finalBounds.right + inset).coerceAtMost(width.toFloat()),
                (finalBounds.bottom + inset).coerceAtMost(height.toFloat())
            )

            // Preserve Apple's flatter SF Pro terminals. Edge softness is
            // supplied by antialiasing and the optical bevel shader, not by
            // rounding the outline path itself.
            val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
                color = Color.WHITE
                style = Paint.Style.FILL
            }
            maskCanvas.drawPath(finalPath, maskPaint)

            // Only the "2" has a subtly softened corner treatment. The small
            // radius rounds its lower foot and terminal transitions without
            // changing the flatter SF Pro ends of 1, 4, 5 or 7.
            val terminalFillet = (fontSize * 0.012f).coerceAtLeast(1f)
            val digitTwoPaint = Paint(maskPaint).apply {
                pathEffect = CornerPathEffect(terminalFillet)
            }
            finalDigitTwoPaths.forEach { twoPath ->
                maskCanvas.drawPath(twoPath, digitTwoPaint)
            }
        } else {
            glyphBounds.setEmpty()
            glyphBaseBounds.setEmpty()
            val emptyColonMask = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            emptyColonMask.eraseColor(Color.TRANSPARENT)
            colonMask = emptyColonMask
            colonMaskOriginX = 0f
            colonMaskOriginY = 0f
        }

        glyphMask = mask
        if (rebuildDistanceField) {
            rebuildGlyphSdf()
        } else {
            // Keep the previous bevel field alive while the new weight mask is
            // previewed. The final accurate SDF is generated once on release.
            glyphSdfStale = true
            updateGlyphMaskInput()
        }
        retireGlyphMask(previousMask)
        retireColonMask(previousColonMask)

        glyphBaseBounds.set(glyphBounds)
        maskClockScale = clockScale
        maskClockWidthScale = clockWidthScale
        gestureOffsetX = 0f
        gestureOffsetY = 0f
        previewScaleX = 1f
        previewScaleY = 1f
        cachedTimeText = timeText
    }

    private fun createColonMask(
        centers: List<Pair<Float, Float>>,
        radius: Float,
        surfaceWidth: Int,
        surfaceHeight: Int
    ): Triple<Bitmap, Float, Float> {
        if (centers.isEmpty()) {
            val empty = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            empty.eraseColor(Color.TRANSPARENT)
            return Triple(empty, 0f, 0f)
        }

        val padding = max(3f, radius * 0.15f)
        val left = (centers.minOf { it.first } - radius - padding)
            .toInt().coerceIn(0, surfaceWidth - 1)
        val top = (centers.minOf { it.second } - radius - padding)
            .toInt().coerceIn(0, surfaceHeight - 1)
        val right = (centers.maxOf { it.first } + radius + padding)
            .toInt().coerceIn(left + 1, surfaceWidth)
        val bottom = (centers.maxOf { it.second } + radius + padding)
            .toInt().coerceIn(top + 1, surfaceHeight)

        val mask = Bitmap.createBitmap(
            right - left,
            bottom - top,
            Bitmap.Config.ARGB_8888
        )
        val maskCanvas = Canvas(mask)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        centers.forEach { (x, y) ->
            maskCanvas.drawCircle(x - left, y - top, radius, paint)
        }
        return Triple(mask, left.toFloat(), top.toFloat())
    }

    private fun rebuildGlyphSdf() {
        val mask = glyphMask
        if (mask == null || mask.isRecycled || glyphBounds.isEmpty) {
            glyphSdfStale = false
            return
        }

        val previousSdf = glyphSdf
        val previousOriginX = glyphSdfOriginX
        val previousOriginY = glyphSdfOriginY
        val previousScale = glyphSdfScale

        val replacement = try {
            createGlyphSdf(mask, glyphBounds)
        } catch (memoryError: OutOfMemoryError) {
            diagnostics.report("glyph-sdf-allocation-out-of-memory", memoryError)
            Log.w(
                TAG,
                "SDF allocation skipped; keeping the previous bevel field.",
                memoryError
            )
            null
        }

        if (replacement != null) {
            glyphSdf = replacement
            glyphSdfStale = false
            updateGlyphMaskInput()
            retireGlyphSdf(previousSdf)
        } else {
            glyphSdf = previousSdf
            glyphSdfOriginX = previousOriginX
            glyphSdfOriginY = previousOriginY
            glyphSdfScale = previousScale
            glyphSdfStale = previousSdf != null
            updateGlyphMaskInput()
        }
    }

    private fun retireGlyphMask(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled || bitmap === glyphMask) return
        retiredGlyphMasks += bitmap
        while (retiredGlyphMasks.size > 1) {
            val oldest = retiredGlyphMasks.removeAt(0)
            if (!oldest.isRecycled) oldest.recycle()
        }
    }

    private fun retireColonMask(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled || bitmap === colonMask) return
        retiredColonMasks += bitmap
        while (retiredColonMasks.size > 1) {
            val oldest = retiredColonMasks.removeAt(0)
            if (!oldest.isRecycled) oldest.recycle()
        }
    }

    private fun retireGlyphSdf(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled || bitmap === glyphSdf) return
        retiredGlyphSdfs += bitmap
        while (retiredGlyphSdfs.size > 1) {
            val oldest = retiredGlyphSdfs.removeAt(0)
            if (!oldest.isRecycled) oldest.recycle()
        }
    }

    /** Widths are optical advances in a single normalized cap-height grid. */
    private fun vectorDigitAdvanceEm(glyph: String): Float = when (glyph) {
        "0" -> 0.43f
        "1" -> 0.38f
        "2" -> 0.39f
        "3" -> 0.39f
        "4" -> 0.42f
        "5" -> 0.39f
        "6" -> 0.41f
        "7" -> 0.39f
        "8" -> 0.43f
        "9" -> 0.41f
        else -> 0.39f
    }

    /**
     * Builds a compact signed-distance field around the current glyph bounds.
     * A two-pass 5/7 chamfer transform closely approximates Euclidean
     * distance and keeps contour normals directionally consistent.
     */
    private fun createGlyphSdf(source: Bitmap, bounds: RectF): Bitmap? {
        if (source.isRecycled || bounds.isEmpty) return null

        val padding = max(
            SDF_PADDING_PX,
            (context.resources.displayMetrics.density * 30f).roundToInt()
        )
        val left = (bounds.left.roundToInt() - padding).coerceIn(0, source.width - 1)
        val top = (bounds.top.roundToInt() - padding).coerceIn(0, source.height - 1)
        val right = (bounds.right.roundToInt() + padding).coerceIn(left + 1, source.width)
        val bottom = (bounds.bottom.roundToInt() + padding).coerceIn(top + 1, source.height)

        val cropWidth = right - left
        val cropHeight = bottom - top
        val cropPixels = cropWidth.toLong() * cropHeight.toLong()
        if (cropPixels <= 0L) return null

        // Keep the preview at native SDF resolution for crisp bevels. Only
        // downsample unusually large full-screen crops to bound temporary memory.
        val maxSdfPixels = 420_000f
        val sdfScale = min(
            1f,
            kotlin.math.sqrt(maxSdfPixels / cropPixels.toFloat())
        ).coerceAtLeast(0.40f)
        val sdfWidth = max(1, (cropWidth * sdfScale).roundToInt())
        val sdfHeight = max(1, (cropHeight * sdfScale).roundToInt())
        val pixelCountLong = sdfWidth.toLong() * sdfHeight.toLong()
        if (pixelCountLong <= 0L || pixelCountLong > 450_000L) return null

        val pixelCount = pixelCountLong.toInt()
        diagnostics.record("SDF allocation start: crop=${cropWidth}x${cropHeight} output=${sdfWidth}x${sdfHeight} pixels=$pixelCount")
        val sourcePixels = IntArray(pixelCount)
        if (sdfScale >= 0.999f) {
            source.getPixels(sourcePixels, 0, sdfWidth, left, top, sdfWidth, sdfHeight)
        } else {
            val sampleBitmap = Bitmap.createBitmap(
                sdfWidth,
                sdfHeight,
                Bitmap.Config.ARGB_8888
            )
            try {
                val sampleCanvas = Canvas(sampleBitmap)
                sampleCanvas.drawBitmap(
                    source,
                    Rect(left, top, right, bottom),
                    Rect(0, 0, sdfWidth, sdfHeight),
                    Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
                )
                sampleBitmap.getPixels(sourcePixels, 0, sdfWidth, 0, 0, sdfWidth, sdfHeight)
            } finally {
                sampleBitmap.recycle()
            }
        }

        val maxUnits = (SDF_RANGE_PX * SDF_CARDINAL_COST.toFloat()).roundToInt()
        val infinity = (maxUnits + 1).toShort()
        val distanceToOutside = ShortArray(pixelCount)
        val distanceToInside = ShortArray(pixelCount)
        for (index in 0 until pixelCount) {
            val inside = Color.alpha(sourcePixels[index]) >= 128
            distanceToOutside[index] = if (inside) infinity else 0
            distanceToInside[index] = if (inside) 0 else infinity
        }

        chamferDistance(distanceToOutside, sdfWidth, sdfHeight, maxUnits)
        chamferDistance(distanceToInside, sdfWidth, sdfHeight, maxUnits)

        val sdfPixels = IntArray(pixelCount)
        for (index in 0 until pixelCount) {
            val alpha = Color.alpha(sourcePixels[index])
            val inside = alpha >= 128
            val signedUnits = if (inside) {
                distanceToOutside[index].toInt()
            } else {
                -distanceToInside[index].toInt()
            }
            // Preserve fractional coverage from the original antialiased edge
            // rather than snapping every contour pixel to fully inside/outside.
            val coverageCorrection = if (alpha in 1..254) {
                (alpha / 255f - 0.5f) / sdfScale
            } else {
                0f
            }
            val signedOutputPixels =
                signedUnits / SDF_CARDINAL_COST.toFloat() / sdfScale +
                    coverageCorrection
            val encoded = (
                0.5f + signedOutputPixels / (2f * SDF_RANGE_PX)
            ).coerceIn(0f, 1f)
            sdfPixels[index] = Color.argb(
                (encoded * 255f).roundToInt(),
                255,
                255,
                255
            )
        }

        return Bitmap.createBitmap(
            sdfWidth,
            sdfHeight,
            Bitmap.Config.ARGB_8888
        ).also { sdf ->
            sdf.setPixels(sdfPixels, 0, sdfWidth, 0, 0, sdfWidth, sdfHeight)
            glyphSdfOriginX = left.toFloat()
            glyphSdfOriginY = top.toFloat()
            glyphSdfScale = sdfScale
            diagnostics.record("SDF allocation complete: $pixelCount pixels")
        }
    }

    private fun chamferDistance(
        distances: ShortArray,
        width: Int,
        height: Int,
        maxUnits: Int
    ) {
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val index = row + x
                var best = distances[index].toInt()
                if (x > 0) {
                    best = min(
                        best,
                        distances[index - 1].toInt() + SDF_CARDINAL_COST
                    )
                }
                if (y > 0) {
                    best = min(
                        best,
                        distances[index - width].toInt() + SDF_CARDINAL_COST
                    )
                    if (x > 0) {
                        best = min(
                            best,
                            distances[index - width - 1].toInt() + SDF_DIAGONAL_COST
                        )
                    }
                    if (x + 1 < width) {
                        best = min(
                            best,
                            distances[index - width + 1].toInt() + SDF_DIAGONAL_COST
                        )
                    }
                }
                distances[index] = min(best, maxUnits).toShort()
            }
        }
        for (y in height - 1 downTo 0) {
            val row = y * width
            for (x in width - 1 downTo 0) {
                val index = row + x
                var best = distances[index].toInt()
                if (x + 1 < width) {
                    best = min(
                        best,
                        distances[index + 1].toInt() + SDF_CARDINAL_COST
                    )
                }
                if (y + 1 < height) {
                    best = min(
                        best,
                        distances[index + width].toInt() + SDF_CARDINAL_COST
                    )
                    if (x > 0) {
                        best = min(
                            best,
                            distances[index + width - 1].toInt() + SDF_DIAGONAL_COST
                        )
                    }
                    if (x + 1 < width) {
                        best = min(
                            best,
                            distances[index + width + 1].toInt() + SDF_DIAGONAL_COST
                        )
                    }
                }
                distances[index] = min(best, maxUnits).toShort()
            }
        }
    }

    private fun buildAdaptiveZeroPath(bounds: RectF): Path {
        val width = bounds.width().coerceAtLeast(1f)
        val height = bounds.height().coerceAtLeast(1f)
        val centerX = bounds.centerX()
        val outer = RectF(
            bounds.left - width * 0.075f,
            bounds.top,
            bounds.right + width * 0.075f,
            bounds.bottom
        )
        val outerWidth = outer.width()

        return Path().apply {
            fillType = Path.FillType.EVEN_ODD
            // Broad vertical sides with rounded shoulders: avoid the narrow,
            // elliptical zero produced by simply stretching a text glyph.
            addRoundRect(
                outer,
                outerWidth * 0.30f,
                height * 0.115f,
                Path.Direction.CW
            )
            val counter = RectF(
                centerX - outerWidth * 0.205f,
                bounds.top + height * 0.145f,
                centerX + outerWidth * 0.205f,
                bounds.bottom - height * 0.145f
            )
            addRoundRect(
                counter,
                outerWidth * 0.16f,
                height * 0.075f,
                Path.Direction.CCW
            )
        }
    }

    private fun buildAdaptiveEightPath(bounds: RectF): Path {
        val left = bounds.left
        val right = bounds.right
        val top = bounds.top
        val bottom = bounds.bottom
        val width = bounds.width().coerceAtLeast(1f)
        val height = bounds.height().coerceAtLeast(1f)
        val centerX = bounds.centerX()
        val waistY = bounds.centerY()

        return Path().apply {
            fillType = Path.FillType.EVEN_ODD

            // A continuous outer silhouette: broad upper/lower bowls with a
            // soft inward waist, keeping the characteristic SF-like rhythm.
            moveTo(centerX, top)
            cubicTo(left + width * 0.22f, top, left, top + height * 0.10f,
                left, top + height * 0.27f)
            cubicTo(left, top + height * 0.40f, left + width * 0.13f,
                top + height * 0.46f, centerX, waistY)
            cubicTo(left + width * 0.13f, top + height * 0.54f, left,
                top + height * 0.61f, left, top + height * 0.75f)
            cubicTo(left, bottom - height * 0.10f, left + width * 0.22f,
                bottom, centerX, bottom)
            cubicTo(right - width * 0.22f, bottom, right,
                bottom - height * 0.10f, right, top + height * 0.75f)
            cubicTo(right, top + height * 0.61f, right - width * 0.13f,
                top + height * 0.54f, centerX, waistY)
            cubicTo(right - width * 0.13f, top + height * 0.46f, right,
                top + height * 0.40f, right, top + height * 0.27f)
            cubicTo(right, top + height * 0.10f, right - width * 0.22f,
                top, centerX, top)
            close()

            // Tall rounded-square counters are the defining detail of the
            // expanded lock-screen 8; EVEN_ODD cuts them from the silhouette.
            val counterLeft = left + width * 0.245f
            val counterRight = right - width * 0.245f
            val counterRadiusX = width * 0.095f
            val counterRadiusY = height * 0.045f
            addRoundRect(
                RectF(
                    counterLeft,
                    top + height * 0.125f,
                    counterRight,
                    top + height * 0.425f
                ),
                counterRadiusX,
                counterRadiusY,
                Path.Direction.CW
            )
            addRoundRect(
                RectF(
                    counterLeft - width * 0.008f,
                    top + height * 0.575f,
                    counterRight + width * 0.008f,
                    bottom - height * 0.115f
                ),
                counterRadiusX,
                counterRadiusY,
                Path.Direction.CW
            )
        }
    }

    /**
     * Small per-digit optical advances remain on top of Apple's true wdth axis.
     * The smooth per-glyph curve preserves the tall lock-screen proportions.
     */
    private fun glyphWidthProfile(glyph: String, expansion: Float): Float {
        val progress = expansion.coerceIn(0f, 1f)
        val compact = when (glyph) {
            "1" -> 1.00f
            "2" -> 1.08f
            "3" -> 1.04f
            "4" -> 1.08f
            "5" -> 1.04f
            "6" -> 1.04f
            "7" -> 0.96f
            "8" -> 1.08f
            "9" -> 1.04f
            "0" -> 1.07f
            else -> 1.00f
        }
        val expanded = when (glyph) {
            "1" -> 1.00f
            "2" -> 1.13f
            "3" -> 1.07f
            "4" -> 1.12f
            "5" -> 1.07f
            "6" -> 1.08f
            "7" -> 0.98f
            "8" -> 1.12f
            "9" -> 1.08f
            "0" -> 1.11f
            else -> 1.00f
        }
        return compact + (expanded - compact) * progress
    }

    private fun glyphHeightFactor(glyph: String): Float {
        val optical = when (glyph) {
            "1" -> 1.015f
            "4", "7" -> 0.985f
            "3", "5" -> 1.01f
            else -> 1.0f
        }
        return optical
    }

    private fun wallpaperWeightAdjustment(
        x: Float,
        y: Float,
        radius: Float
    ): Int {
        val image = backdrop ?: return 0
        if (image.isRecycled || image.width <= 0 || image.height <= 0) return 0

        val centerX = x.roundToInt().coerceIn(0, image.width - 1)
        val centerY = y.roundToInt().coerceIn(0, image.height - 1)
        val offset = radius.roundToInt().coerceAtLeast(1)
        var luminance = 0f
        var samples = 0

        for (dy in -1..1) {
            for (dx in -1..1) {
                val sx = (centerX + dx * offset).coerceIn(0, image.width - 1)
                val sy = (centerY + dy * offset).coerceIn(0, image.height - 1)
                val pixel = image.getPixel(sx, sy)
                luminance += (
                    Color.red(pixel) * 0.2126f +
                        Color.green(pixel) * 0.7152f +
                        Color.blue(pixel) * 0.0722f
                    ) / 255f
                samples++
            }
        }

        val average = luminance / samples.coerceAtLeast(1)
        return when {
            average > 0.70f -> 28
            average > 0.54f -> 14
            average < 0.14f -> 8
            else -> 0
        }
    }

    private fun clockFontWidthCoordinate(): Int = when (clockFontStyle) {
        1 -> SF_PRO_WIDTH_CONDENSED
        2 -> SF_PRO_WIDTH_COMPRESSED
        3 -> SF_PRO_WIDTH_EXPANDED
        else -> SF_PRO_WIDTH_REGULAR
    }

    private fun typefaceFor(
        weight: Int,
        opticalSize: Int,
        widthProfile: Int
    ): Typeface {
        val safeWeight = weight.coerceIn(100, 900)
        val safeWidth = widthProfile.coerceIn(SF_PRO_WIDTH_COMPRESSED, SF_PRO_WIDTH_EXPANDED)
        val cacheKey = safeWeight * 1000 + safeWidth
        typefaceCache[cacheKey]?.let { return it }

        val variableFace = try {
            Typeface.Builder(context.assets, VARIABLE_FONT_ASSET_PATH)
                .setFontVariationSettings(
                    "'wght' $safeWeight, 'wdth' $safeWidth"
                )
                .build()
        } catch (fontException: Exception) {
            diagnostics.record("SF Pro renderer font fallback: ${fontException.javaClass.simpleName}")
            Log.w(
                TAG,
                "Official Apple SF Pro variable font unavailable; using its Display Thin face.",
                fontException
            )
            fallbackTypeface
        } catch (memoryError: OutOfMemoryError) {
            diagnostics.report("renderer-sf-pro-typeface-allocation", memoryError)
            fallbackTypeface
        }

        if (typefaceCache.size >= 4) {
            typefaceCache.clear()
        }
        typefaceCache[cacheKey] = variableFace
        return variableFace
    }

    private fun updateGlyphMaskInput() {
        val shader = glassShader ?: return
        val mask = glyphMask ?: return
        try {
            shader.setInputShader(
                "uGlyphMask",
                BitmapShader(mask, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            )
            val colon = colonMask ?: Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            shader.setInputShader(
                "uColonMask",
                BitmapShader(colon, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            )
            shader.setFloatUniform("uColonOrigin", colonMaskOriginX, colonMaskOriginY)
            val sdf = glyphSdf ?: mask
            shader.setInputShader(
                "uGlyphSdf",
                BitmapShader(sdf, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            )
            shader.setFloatUniform("uSdfOrigin", glyphSdfOriginX, glyphSdfOriginY)
            shader.setFloatUniform("uSdfScale", glyphSdfScale)
            shader.setFloatUniform("uSdfRange", SDF_RANGE_PX)
        } catch (exception: RuntimeException) {
            diagnostics.report("shader-glyph-input-refresh", exception)
            Log.w(TAG, "Could not refresh glyph mask/SDF inputs.", exception)
            glassShader = null
        }
    }

    private fun extractWallpaperAccent(bitmap: Bitmap): Int {
        // Pick the strongest recurring saturated hue as the wallpaper's
        // secondary accent. Near-white and near-black samples are ignored.
        val bins = 24
        val weights = DoubleArray(bins)
        val red = DoubleArray(bins)
        val green = DoubleArray(bins)
        val blue = DoubleArray(bins)
        val hsv = FloatArray(3)
        val stepX = max(1, bitmap.width / 72)
        val stepY = max(1, bitmap.height / 144)

        var y = 0
        while (y < bitmap.height) {
            var x = 0
            while (x < bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                Color.colorToHSV(pixel, hsv)
                val saturation = hsv[1]
                val value = hsv[2]
                if (saturation >= 0.28f && value >= 0.22f && value <= 0.98f) {
                    val bin = ((hsv[0] / 360f) * bins)
                        .toInt()
                        .coerceIn(0, bins - 1)
                    val weight = saturation.toDouble() *
                        (0.45 + value.toDouble() * 0.55)
                    weights[bin] += weight
                    red[bin] += Color.red(pixel) * weight
                    green[bin] += Color.green(pixel) * weight
                    blue[bin] += Color.blue(pixel) * weight
                }
                x += stepX
            }
            y += stepY
        }

        var best = 0
        for (index in 1 until bins) {
            if (weights[index] > weights[best]) best = index
        }
        if (weights[best] <= 0.0) return Color.rgb(145, 150, 160)

        return Color.rgb(
            (red[best] / weights[best]).toInt().coerceIn(0, 255),
            (green[best] / weights[best]).toInt().coerceIn(0, 255),
            (blue[best] / weights[best]).toInt().coerceIn(0, 255)
        )
    }

    private fun rebuildGlassShader() {
        val background = wallpaperSample ?: return
        val mask = glyphMask ?: return

        try {
            val shader = RuntimeShader(AGSL)
            shader.setInputShader(
                "uWallpaper",
                BitmapShader(
                    background,
                    Shader.TileMode.CLAMP,
                    Shader.TileMode.CLAMP
                )
            )
            shader.setInputShader(
                "uReflectionEnvironment",
                BitmapShader(
                    background,
                    Shader.TileMode.CLAMP,
                    Shader.TileMode.CLAMP
                )
            )
            shader.setInputShader(
                "uGlyphMask",
                BitmapShader(
                    mask,
                    Shader.TileMode.CLAMP,
                    Shader.TileMode.CLAMP
                )
            )
            val colon = colonMask ?: Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
            shader.setInputShader(
                "uColonMask",
                BitmapShader(colon, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            )
            shader.setFloatUniform("uColonOrigin", colonMaskOriginX, colonMaskOriginY)
            val sdf = glyphSdf ?: mask
            shader.setInputShader(
                "uGlyphSdf",
                BitmapShader(sdf, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
            )
            shader.setFloatUniform("uSdfOrigin", glyphSdfOriginX, glyphSdfOriginY)
            shader.setFloatUniform("uSdfScale", glyphSdfScale)
            shader.setFloatUniform("uSdfRange", SDF_RANGE_PX)
            shader.setFloatUniform(
                "uResolution",
                surfaceWidth.toFloat(),
                surfaceHeight.toFloat()
            )
            shader.setFloatUniform(
                "uAccentColor",
                Color.red(adaptiveAccentColor) / 255f,
                Color.green(adaptiveAccentColor) / 255f,
                Color.blue(adaptiveAccentColor) / 255f
            )
            shader.setFloatUniform(
                "uClockTint",
                Color.red(clockColor) / 255f,
                Color.green(clockColor) / 255f,
                Color.blue(clockColor) / 255f
            )
            shader.setFloatUniform(
                "uTintStrength",
                if (clockColor == Color.WHITE) 0f else 0.16f
            )
            shader.setFloatUniform("uFrosted", if (glassMode) 0f else 1f)
            shader.setFloatUniform(
                "uClockCenter",
                clockCenterX,
                clockCenterY
            )
            glassShader = shader
        } catch (exception: RuntimeException) {
            diagnostics.report("agsl-shader-initialization", exception)
            Log.w(TAG, "Unable to initialize AGSL shader.", exception)
            glassShader = null
        } catch (memoryError: OutOfMemoryError) {
            diagnostics.report("agsl-shader-out-of-memory", memoryError)
            glassShader = null
        }
    }

    /**
     * Re-reads the injected clock source and rebuilds glyph outlines when its
     * sample changes. Fixed sample state remains inside the Digit Lab host.
     */
    fun refreshClockText() {
        cachedTimeText = null
        if (surfaceWidth > 0 && surfaceHeight > 0) rebuildGlyphMask(force = true)
    }

    private fun currentTimeText(): String = clockTextSource.currentTimeText(context)

    private fun loadFallbackTypeface(): Typeface {
        return try {
            Typeface.createFromAsset(context.assets, FONT_ASSET_PATH)
        } catch (exception: Exception) {
            Log.w(TAG, "SF Pro Display Thin unavailable; using system light font.", exception)
            Typeface.create("sans-serif-light", Typeface.NORMAL)
        }
    }
}
