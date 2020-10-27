#version 300 es
precision mediump float;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec4 u_mouse;
uniform vec3 u_palette[8];
uniform float u_shiny[8];

const float pi = 3.141592653589793;
const float tau = pi * 2.0;
const float hpi = pi * 0.5;
const float phi = (1.0+sqrt(5.0))/2.0;

out vec4 outColor;


#define MAX_STEPS 100
#define MAX_DIST 100.
#define SURF_DIST .001

#define ROT(a) mat2(cos(a), -sin(a), sin(a), cos(a))
#define SHEARX(a) mat2(1, 0, sin(a), 1)

float rand(float n){return fract(sin(n) * 43758.5453123);}

////////////////////// NOISE

//	Simplex 3D Noise
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
    dot(p2,x2), dot(p3,x3) ) );
}


// Camera helper

vec3 Camera(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
    r = normalize(
    cross(
    vec3(0, 1, 0),
    f
    )
    ),
    u = cross(f, r),
    c = p + f * z,
    i = c + uv.x*r + uv.y*u,
    d = normalize(i-p);
    return d;
}


// 2d rotation matrix helper
mat2 Rot(float a) {
    float x = cos(a);
    float y = sin(a);
    return mat2(x, -y, y, x);
}

// RAY MARCHING PRIMITIVES

float smin(float a, float b, float k) {
    float h = clamp(0.5+0.5*(b-a)/k, 0., 1.);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    return length(p-c)-r;
}

float sdCylinder(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b-a;
    vec3 ap = p-a;

    float t = dot(ab, ap) / dot(ab, ab);
    //t = clamp(t, 0., 1.);

    vec3 c = a + t*ab;

    float x = length(p-c)-r;
    float y = (abs(t-.5)-.5)*length(ab);
    float e = length(max(vec2(x, y), 0.));
    float i = min(max(x, y), 0.);

    return e+i;
}

float sdCappedCylinder( vec3 p, float h, float r )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdSphere(vec3 p, float s)
{
    return length(p)-s;
}

float sdTorus(vec3 p, vec2 r) {
    float x = length(p.xz)-r.x;
    return length(vec2(x, p.y))-r.y;
}

float sdRoundBox(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}


float sdBeam(vec3 p, vec3 c)
{
    return length(p.xz - c.xy)-c.z;
}

float dBox(vec3 p, vec3 s) {
    p = abs(p)-s;
    return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
}

vec2 opUnion(vec2 curr, float d, float id)
{
//    if (d < curr.x)
//    {
//        curr.x = d;
//        curr.y = id;
//    }
//
//    return curr;

    float s = step(curr.x, d);
    return s * curr + (1.0 - s) * vec2(d, id);
}


// Minimum - with corresponding object ID.
vec2 objMin(vec2 a, vec2 b){

    // Returning the minimum distance along with the ID of the
    // object. This is one way to do it. There are others.

    // Equivalent to: return a.x < b.x ? a: b;
    float s = step(a.x, b.x);
    return s*a + (1. - s)*b;
}


vec2 softMinUnion(vec2 curr, float d, float id)
{
    if (d < curr.x)
    {
        curr.x = smin(curr.x, d, 0.5);
        curr.y = id;
    }

    return curr;
}


float sdBoundingBox(vec3 p, vec3 b, float e)
{
    p = abs(p)-b;
    vec3 q = abs(p+e)-e;
    return min(min(
    length(max(vec3(p.x, q.y, q.z), 0.0))+min(max(p.x, max(q.y, q.z)), 0.0),
    length(max(vec3(q.x, p.y, q.z), 0.0))+min(max(q.x, max(p.y, q.z)), 0.0)),
    length(max(vec3(q.x, q.y, p.z), 0.0))+min(max(q.x, max(q.y, p.z)), 0.0));
}

float sdHexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
    length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
    p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float shape(float v, float x)
{
    return x > 0.0 ? -abs(v) : abs(v);
}

const mat2 frontPlaneRot = ROT(0.05235987755982988);
const mat2 backPlaneRot = ROT(-0.05235987755982988);
const mat2 sCutRot = ROT(0.88);
const mat2 rotate90 = ROT(1.5707963267948966);
const mat2 fourShear = SHEARX(-0.20943951023931953);

vec2 getDistance(vec3 p) {

    float t = u_time;


    // ground plane
    //float pd = p.y + 2.0;

    vec2 result = vec2(1e6, 0);
    float sphere = sdSphere( p , 1.8 );

    if (sphere < 0.2)
    {
        vec3 cutoutPos = p;

        cutoutPos.yz *= rotate90;

        float cutout = sdCappedCylinder(cutoutPos, 1.89, 1.0);

        float gCutout = sdCappedCylinder(cutoutPos, 1.78, 1.0);

        float gGapCutout = dBox(cutoutPos - vec3(1.36,0, 0.36), vec3(1,1,0.36));

        float gInnerCutout = sdRoundBox(cutoutPos - vec3(-0.2,0,0), vec3(0.42,1,1.12), 0.45);

        float flattenedSideCutout = dot(p - vec3(-0.98,0,0), vec3(-1,0,0) );

        float gFlapCutout = dBox(cutoutPos - vec3(0.62 + 0.35,0, -0.35), vec3(0.5,1,0.35));

        float round = 0.4;

        // Letter S

        float sCutout = sdRoundBox(cutoutPos - vec3(-0.2,0,0.5), vec3(0.44 - round, 1.0- round, 0.8-round), round);


        round = 0.2;

        vec3 s2Pos = cutoutPos- vec3( 0.21,0,0.56);


        s2Pos.xz *= sCutRot;

        float s2Cutout = sdRoundBox(s2Pos , vec3(0.51 - round, 1.1 - round, 0.18-round), round);
        sCutout = max(-s2Cutout, sCutout);

        s2Pos = cutoutPos- vec3( -0.6,0,0.36);
        s2Pos.xz *= sCutRot;

        float s3Cutout = sdRoundBox(s2Pos , vec3(0.51 - round, 1.1 - round, 0.18-round), round);
        sCutout = max(-s3Cutout, sCutout);

        float sSerif = dBox(cutoutPos - vec3(0.2,0, 1.02), vec3(0.05,1,0.24));
        sCutout = min(sSerif, sCutout);

        sSerif = dBox(cutoutPos - vec3(-0.66,0, 0.03), vec3(0.05,1,0.24));
        sCutout = min(sSerif, sCutout);

        // Zero


        vec3 elongation = vec3(0,1,0.32);

        vec3 zCutoutPos = cutoutPos- vec3(-0.47,0,-0.95);

        vec3 q = zCutoutPos - clamp( zCutoutPos, -elongation, elongation );

        float zeroCutout = sdTorus(q , vec2(0.155, 0.07));


        // four

        vec3 fourShBoxPos = p;
        fourShBoxPos.yx *= fourShear;
        float fourShBox = dBox(fourShBoxPos - vec3(.14,-0.8,0), vec3(0.1,0.38,1));


        float fourHBox = dBox(p - vec3(.14,-1.08,0), vec3(0.28,0.1,1));

        float fourVBox = dBox(p - vec3(.25,-1.15,0), vec3(0.1, 0.3,1));

        gInnerCutout = max(flattenedSideCutout, gInnerCutout);
        gInnerCutout = max(-gFlapCutout, gInnerCutout);
        gInnerCutout = max(-sCutout, gInnerCutout);
        gInnerCutout = max(-zeroCutout, gInnerCutout);
        gInnerCutout = max(-fourShBox, gInnerCutout);
        gInnerCutout = max(-fourHBox, gInnerCutout);
        gInnerCutout = max(-fourVBox, gInnerCutout);

        gCutout = max(-gGapCutout, gCutout);
        gCutout = max(-gInnerCutout, gCutout);
        cutout = max(-gCutout, cutout);

        vec3 frontPlane = vec3(0,0,-1);
        frontPlane.yz *= frontPlaneRot;

        vec3 backPlane = vec3(0,0,1);
        backPlane.yz *= backPlaneRot;

        float front = dot(p + vec3(0,0,.25), frontPlane );
        float back = dot(p - vec3(0,0,.25), backPlane)  ;

        sphere = max(sphere, front);
        sphere = max(sphere, back) - 0.2;
        sphere = max(-cutout, sphere);

        // DEBUG
        //    result = opUnion(result, gFlapCutout, 2.1);
        //    result = opUnion(result, gGapCutout, 1.9);
        //    result = opUnion(result, sCutout, 2.2);
        //    result = opUnion(result, zeroCutout, 2.3);
        //    result = opUnion(result, fourShBox, 2.4);
        //    result = opUnion(result, fourHBox, 2.5);
        //    result = opUnion(result, fourVBox, 2.6);
        //    result = opUnion(result, s2Cutout, 2.7);
        //    result = opUnion(result, s3Cutout, 2.8);

    }

    //result = opUnion(result, pd, 4.0);
    result = opUnion(result, sphere, 1.0);


    vec3 shaftPos = p;

    shaftPos.x += (sin((p.z + u_time * 10.0) * 0.23) + sin((p.z + u_time * 10.0) * 0.27)) * 0.55;

    shaftPos.yz *= rotate90;
    float zOffset = - u_time * 10.0;

    float shaft = -sdBeam( shaftPos, vec3(0,0,3) );

    float mat = 4.0;

    float d = length(vec3(p.x, p.y, 0.0));
    vec3 sideShaftPos = p;

    if (abs(d) > 1.9)
    {
        vec3 noisePos = p - vec3(0,0, zOffset);
        float n = snoise(noisePos);
        float off = shape(n,p.x) * 0.25;
        shaft = -sdBeam( shaftPos + off , vec3(0,0,3) ) * 0.6;
        mat = n < 0.4 ? 4.0 : 5.0;

        sideShaftPos += off * 0.9;

        float bottom = dot(p + vec3(0,2.2,0) + off * 0.5, vec3(0,1,0) );

        result = opUnion(result, bottom, 3.0);
    }


    sideShaftPos.z += u_time * 10.0;


    sideShaftPos.x += floor(rand(floor(sideShaftPos.z/60.0)) * 4.0) * 10.0 - 10.0;
    sideShaftPos.z = mod(sideShaftPos.z - 10. + 0.5 * 60.0, 60.0)- 30.0;

    float sideShaft = -dBox( sideShaftPos, vec3(10,3,3) );

    shaft = max(shaft, sideShaft);

    result = opUnion(result, shaft, mat);


    vec3 supportPos = p;
    supportPos.x = abs(supportPos.x);
    supportPos.x -= 2.5;
    supportPos.z -= zOffset;

    float c = 20.0;
    supportPos.z = mod(supportPos.z+0.5*c,c)-0.5*c;

    supportPos.yz *= rotate90;


    float support = sdHexPrism(supportPos, vec2(0.4, 2.5)) - 0.05;

    vec3 support2Pos = p;

    support2Pos.z = mod(supportPos.z+0.5*c,c)-0.5*c;

    float support2 = sdHexPrism(support2Pos - vec3(0,2.5,0), vec2(0.4, 20)) - 0.05;

    vec3 support3Pos = p;
    support3Pos.z -= zOffset;
    support3Pos.z = mod(support3Pos.z+0.5*c,c)-0.5*c;
    support3Pos.xz *= rotate90;

    float support3 = sdHexPrism(support3Pos - vec3(0,2.5,0), vec2(0.4, 20)) - 0.05;

    result = opUnion(result, support, 6.0);
    result = opUnion(result, support2, 6.0);
    result = opUnion(result, support3, 6.0);


    return result;
}


vec2 rayMarch(vec3 ro, vec3 rd) {


    float dO = 0.;
    float id = 0.0;

    for (int i=0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd*dO;
        vec2 result = getDistance(p);
        float dS = result.x;
        dO += dS;
        id = result.y;
        if (dO > MAX_DIST || abs(dS) < SURF_DIST)
        break;
    }

    return vec2(dO, id);
}

vec3 getNormal(vec3 p) {
    float d = getDistance(p).x;
    vec2 e = vec2(.001, 0);

    vec3 n = d - vec3(
        getDistance(p-e.xyy).x,
        getDistance(p-e.yxy).x,
        getDistance(p-e.yyx).x
    );

    return normalize(n);
}


vec3 getPaletteColor(float id)
{
    int last = u_palette.length() - 1;
    //return id < float(last) ? mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id)) : u_palette[last];
    return mix(u_palette[int(id)], u_palette[int(id) + 1], fract(id));
}


vec3 applyFog( in vec3  rgb,      // original color of the pixel
    in float distance, // camera to point distance
    in vec3  rayOri,   // camera position
    in vec3  rayDir,
    in vec3 p)  // camera to point vector
{
    float pos = p.z + u_time * 12.0;

    float c = 0.008;
    float b = 0.95 + sin((pos + p.x * sin(pos * 0.27)) * 0.31 ) * 0.15 + sin(pos * 0.17 ) * 0.15;

    float fogAmount = c * exp(-rayOri.y*b) * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;
    vec3  fogColor  = #004d9d;
    return mix( rgb, fogColor, fogAmount );
}
void main(void)
{
    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;
    vec2 m = u_mouse.xy/u_resolution.xy;

    vec3 col = vec3(0);
    vec3 ro = vec3(
    (cos(u_time * 1.7) + cos(u_time * 1.5)) * 0.7,
    (sin(u_time * 2.1) - sin(u_time * 1.9)) * 0.5,
        -10.0 + sin(u_time) * 2.0
    );

    //    ro.yz *= Rot((-m.y + 0.5)* 7.0);
//    ro.xz *= Rot((-m.x + 0.5)* 7.0 + u_time);


    vec3 lookAt = vec3(0);

    vec3 rd = Camera(uv, ro, lookAt, 1.3);

    vec2 result = rayMarch(ro, rd);

    float d = result.x;

    vec3 p = ro + rd * d;
    if (d < MAX_DIST) {

        vec3 lightPos = ro + vec3(0,1,0);
        vec3 lightDir = normalize(lightPos - p);
        vec3 norm = getNormal(p);

        vec3 lightColor = vec3(2.8);

        float id = result.y;

        // ambient
        vec3 ambient = lightColor * vec3(0.001);

        // diffuse
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 tone = getPaletteColor(id);

        if (id == 4.0)
        {
            tone *= snoise(p + vec3(0,0, u_time * 10.0)) * 0.5;
        }

        vec3 diffuse = lightColor * (diff * tone);

        // specular
        vec3 viewDir = normalize(ro);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), u_shiny[int(id)]);
        vec3 specular = lightColor * spec * vec3(0.7843,0.8823,0.9451) * (id == 1.0 ? 0.24 : 0.1);

        col = (ambient + diffuse + specular);

    }
    col = applyFog(col, d, ro, rd, p);


    //col = pow(col, vec3(1.0/2.2));

    outColor = vec4(
        col,
        1.0
    );

    //outColor = vec4(1,0,1,1);
}
