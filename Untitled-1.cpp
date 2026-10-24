#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
using namespace std;

// ============================ 1. Math ============================
struct Vec3
{
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
Vec3 operator-(Vec3 a) { return Vec3(-a.x, -a.y, -a.z); }
Vec3 operator*(Vec3 a, Vec3 b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
Vec3 operator*(double t, Vec3 a) { return Vec3(t * a.x, t * a.y, t * a.z); }
Vec3 operator/(Vec3 a, double t) { return Vec3(a.x / t, a.y / t, a.z / t); }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double length(Vec3 a) { return sqrt(dot(a, a)); }
Vec3 cross(Vec3 a, Vec3 b) { return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
Vec3 unit(Vec3 a) { return a / length(a); }
Vec3 reflect(Vec3 v, Vec3 n) { return v - 2 * dot(v, n) * n; }
Vec3 vmin(Vec3 a, Vec3 b) { return Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
Vec3 vmax(Vec3 a, Vec3 b) { return Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
double luminance(Vec3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

thread_local mt19937 rng;
thread_local long long rayCount = 0; // rays traced by this thread (for statistics)
double rnd() { return uniform_real_distribution<double>(0, 1)(rng); }
double rnd(double a, double b) { return a + (b - a) * rnd(); }
Vec3 randomUnit()
{
    while (true)
    {
        Vec3 p(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1));
        double len2 = dot(p, p);
        if (len2 < 1 && len2 > 1e-8)
            return p / sqrt(len2);
    }
}
Vec3 randomInDisk()
{
    while (true)
    {
        Vec3 p(rnd(-1, 1), rnd(-1, 1), 0);
        if (dot(p, p) < 1)
            return p;
    }
}

// ============================ 2. Scene objects ============================
enum Type
{
    DIFFUSE,
    METAL,
    GLASS,
    LIGHT
};
struct Material
{
    Type type;
    Vec3 color;
    double param; // metal: roughness, glass: index of refraction
    Material(Type t = DIFFUSE, Vec3 c = Vec3(), double p = 0) : type(t), color(c), param(p) {}
};

struct Object
{                 // sphere or triangle
    Vec3 a, b, c; // sphere: a = center | triangle: corners
    double radius;
    bool isTriangle;
    Material mat;
    Vec3 lo, hi, cen; // bounding box + centroid (filled by finalize())
};
Object sphere(Vec3 center, double r, Material m)
{
    Object o;
    o.a = center;
    o.radius = r;
    o.isTriangle = false;
    o.mat = m;
    return o;
}
Object triangle(Vec3 a, Vec3 b, Vec3 c, Material m)
{
    Object o;
    o.a = a;
    o.b = b;
    o.c = c;
    o.radius = 0;
    o.isTriangle = true;
    o.mat = m;
    return o;
}
void finalize(Object &o)
{
    if (!o.isTriangle)
    {
        Vec3 r(o.radius, o.radius, o.radius);
        o.lo = o.a - r;
        o.hi = o.a + r;
    }
    else
    {
        const double pad = 1e-4; // flat triangles need a little thickness
        Vec3 p(pad, pad, pad);
        o.lo = vmin(o.a, vmin(o.b, o.c)) - p;
        o.hi = vmax(o.a, vmax(o.b, o.c)) + p;
    }
    o.cen = (o.lo + o.hi) / 2;
}

struct Ray
{
    Vec3 o, d;
};
struct Hit
{
    double t;
    Vec3 p, n;
    bool front;
    const Material *mat;
};
const double EPS = 0.001;

bool hitObject(const Object &o, const Ray &r, double tmax, Hit &h)
{
    double t;
    Vec3 n;
    if (!o.isTriangle)
    {
        Vec3 oc = r.o - o.a;
        double A = dot(r.d, r.d), B = dot(oc, r.d), C = dot(oc, oc) - o.radius * o.radius;
        double disc = B * B - A * C;
        if (disc < 0)
            return false;
        double sq = sqrt(disc);
        t = (-B - sq) / A;
        if (t < EPS || t > tmax)
        {
            t = (-B + sq) / A;
            if (t < EPS || t > tmax)
                return false;
        }
        n = (r.o + t * r.d - o.a) / o.radius;
    }
    else
    { // Moller-Trumbore
        Vec3 e1 = o.b - o.a, e2 = o.c - o.a, p = cross(r.d, e2);
        double det = dot(e1, p);
        if (fabs(det) < 1e-12)
            return false;
        double invDet = 1 / det;
        Vec3 s = r.o - o.a, q = cross(s, e1);
        double u = dot(s, p) * invDet, v = dot(r.d, q) * invDet;
        t = dot(e2, q) * invDet;
        if (u < 0 || v < 0 || u + v > 1 || t < EPS || t > tmax)
            return false;
        n = unit(cross(e1, e2));
    }
    h.t = t;
    h.p = r.o + t * r.d;
    h.front = dot(r.d, n) < 0;
    h.n = h.front ? n : -n;
    h.mat = &o.mat;
    return true;
}

// ============================ 3. BVH (binned surface-area heuristic) ============================
vector<Object> objects;
struct Node
{
    Vec3 lo, hi;
    int left, right;  // inner node: children
    int first, count; // leaf node: objects[first .. first+count)
    int axis;         // split axis (used to visit the nearer child first)
    Node() : left(-1), right(-1), first(0), count(0), axis(0) {}
};
vector<Node> nodes;
const int MAX_LEAF = 4;

struct AABB
{
    Vec3 lo, hi;
    AABB() : lo(1e30, 1e30, 1e30), hi(-1e30, -1e30, -1e30) {}
    void grow(Vec3 l, Vec3 h)
    {
        lo = vmin(lo, l);
        hi = vmax(hi, h);
    }
    double area() const
    {
        Vec3 d = hi - lo;
        return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

int build(int start, int end)
{
    int id = nodes.size();
    nodes.push_back(Node());
    int n = end - start;

    AABB box, cbox;
    for (int i = start; i < end; i++)
    {
        box.grow(objects[i].lo, objects[i].hi);
        cbox.grow(objects[i].cen, objects[i].cen);
    }
    nodes[id].lo = box.lo;
    nodes[id].hi = box.hi;

    Vec3 ext = cbox.hi - cbox.lo;
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    nodes[id].axis = axis;

    auto makeLeaf = [&]()
    {
        nodes[id].first = start;
        nodes[id].count = n;
        return id;
    };
    if (n <= 2)
        return makeLeaf();

    int mid = -1;
    if (ext[axis] > 1e-9)
    {
        const int B = 12;
        AABB bb[B];
        int cnt[B] = {0};
        auto binOf = [&](const Object &o)
        {
            return min(B - 1, int((o.cen[axis] - cbox.lo[axis]) / ext[axis] * B));
        };
        for (int i = start; i < end; i++)
        {
            int b = binOf(objects[i]);
            cnt[b]++;
            bb[b].grow(objects[i].lo, objects[i].hi);
        }
        // sweep from the right to get the right-side cost of each split, then from the left
        double rightCost[B];
        AABB acc;
        int c = 0;
        for (int i = B - 1; i > 0; i--)
        {
            acc.grow(bb[i].lo, bb[i].hi);
            c += cnt[i];
            rightCost[i] = c ? acc.area() * c : 1e30;
        }
        AABB lacc;
        int lc = 0;
        double best = 1e30;
        int bestSplit = -1;
        for (int i = 0; i < B - 1; i++)
        {
            lacc.grow(bb[i].lo, bb[i].hi);
            lc += cnt[i];
            if (lc == 0 || lc == n)
                continue;
            double cost = lacc.area() * lc + rightCost[i + 1];
            if (cost < best)
            {
                best = cost;
                bestSplit = i;
            }
        }
        if (bestSplit >= 0)
        {
            double splitCost = 1 + best / box.area(); // 1 = cost of visiting the node itself
            if (n <= MAX_LEAF && splitCost >= n)
                return makeLeaf();
            Object *m = partition(&objects[start], &objects[end - 1] + 1,
                                  [&](const Object &o)
                                  { return binOf(o) <= bestSplit; });
            mid = int(m - &objects[0]);
        }
    }
    if (mid <= start || mid >= end)
    { // degenerate: fall back to a median split
        if (n <= MAX_LEAF && ext[axis] <= 1e-9)
            return makeLeaf();
        mid = (start + end) / 2;
        nth_element(objects.begin() + start, objects.begin() + mid, objects.begin() + end,
                    [axis](const Object &a, const Object &b)
                    { return a.cen[axis] < b.cen[axis]; });
    }
    int l = build(start, mid);
    int r = build(mid, end);
    nodes[id].left = l;
    nodes[id].right = r;
    return id;
}

inline bool hitBox(const Node &n, const Ray &r, const Vec3 &inv, double tmax)
{ // slab test
    double t0 = EPS, t1 = tmax;
    for (int i = 0; i < 3; i++)
    {
        double a = (n.lo[i] - r.o[i]) * inv[i], b = (n.hi[i] - r.o[i]) * inv[i];
        t0 = max(t0, min(a, b));
        t1 = min(t1, max(a, b));
        if (t1 < t0)
            return false;
    }
    return true;
}

bool hitBVH(const Ray &r, double &tmax, Hit &h, bool anyHit)
{
    Vec3 inv(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
    int stack[64], sp = 0;
    stack[sp++] = 0;
    bool hit = false;
    while (sp)
    {
        const Node &n = nodes[stack[--sp]];
        if (!hitBox(n, r, inv, tmax))
            continue;
        if (n.count > 0)
        {
            for (int i = n.first; i < n.first + n.count; i++)
                if (hitObject(objects[i], r, tmax, h))
                {
                    hit = true;
                    tmax = h.t;
                    if (anyHit)
                        return true;
                }
        }
        else if (r.d[n.axis] < 0)
        { // visit the nearer child first (pushed last = popped first)
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
        else
        {
            stack[sp++] = n.right;
            stack[sp++] = n.left;
        }
    }
    return hit;
}

bool useBVH = true;
bool hitWorld(const Ray &r, Hit &h, double tmax = 1e30, bool anyHit = false)
{
    rayCount++;
    if (useBVH)
        return hitBVH(r, tmax, h, anyHit);
    bool hit = false;
    for (size_t i = 0; i < objects.size(); i++)
        if (hitObject(objects[i], r, tmax, h))
        {
            hit = true;
            tmax = h.t;
            if (anyHit)
                return true;
        }
    return hit;
}

// ============================ 4. Lights + path tracing ============================
struct Light
{ // rectangular area light
    Vec3 corner, e1, e2, color, normal;
    double area;
};
vector<Light> lights;

// Direct lighting at a diffuse hit: pick a random point on a light and check if it is visible.
Vec3 sampleLights(const Hit &h)
{
    const Light &L = lights[size_t(rnd() * lights.size()) % lights.size()];
    Vec3 p = L.corner + rnd() * L.e1 + rnd() * L.e2;
    Vec3 to = p - h.p;
    double dist2 = dot(to, to), dist = sqrt(dist2);
    Vec3 wi = to / dist;
    double cosS = dot(h.n, wi), cosL = fabs(dot(L.normal, wi)); // light emits on both faces
    if (cosS <= 0 || cosL <= 1e-6)
        return Vec3();
    Hit tmp;
    Ray shadow;
    shadow.o = h.p;
    shadow.d = wi;
    if (hitWorld(shadow, tmp, dist - 0.01, true))
        return Vec3(); // blocked
    double k = cosS * cosL * L.area / (3.14159265358979 * dist2) * lights.size();
    return k * L.color; // (still to be multiplied by the surface color)
}

Vec3 trace(Ray ray, bool useSky)
{
    Vec3 color, weight(1, 1, 1);
    bool lastSpecular = true; // did the last bounce come from a mirror/glass (or the camera)?

    for (int depth = 0; depth < 16; depth++)
    {
        Hit h;
        if (!hitWorld(ray, h))
        {
            if (useSky)
            {
                double t = 0.5 * (unit(ray.d).y + 1);
                color = color + weight * ((1 - t) * Vec3(1, 1, 1) + t * Vec3(0.5, 0.7, 1));
            }
            break;
        }
        const Material &m = *h.mat;
        if (m.type == LIGHT)
        {
            // Diffuse surfaces already sampled this light directly, so don't count it twice.
            if (lastSpecular || lights.empty())
                color = color + weight * m.color;
            break;
        }

        Vec3 dir;
        if (m.type == DIFFUSE)
        {
            if (!lights.empty())
                color = color + weight * m.color * sampleLights(h);
            dir = h.n + randomUnit();
            if (dot(dir, dir) < 1e-12)
                dir = h.n;
            lastSpecular = false;
        }
        else if (m.type == METAL)
        {
            dir = reflect(unit(ray.d), h.n) + m.param * randomUnit();
            if (dot(dir, h.n) <= 0)
                break;
            lastSpecular = true;
        }
        else
        { // GLASS: refract, or reflect (Schlick approximation)
            double ratio = h.front ? 1 / m.param : m.param;
            Vec3 u = unit(ray.d);
            double c = min(dot(-u, h.n), 1.0), s = sqrt(1 - c * c);
            double r0 = pow((1 - ratio) / (1 + ratio), 2);
            if (ratio * s > 1 || r0 + (1 - r0) * pow(1 - c, 5) > rnd())
            {
                dir = reflect(u, h.n);
            }
            else
            {
                Vec3 perp = ratio * (u + c * h.n);
                dir = perp - sqrt(fabs(1 - dot(perp, perp))) * h.n;
            }
            lastSpecular = true;
        }
        if (m.type != GLASS)
            weight = weight * m.color;
        ray.o = h.p;
        ray.d = dir;

        if (depth > 3)
        { // Russian roulette
            double p = min(max(weight.x, max(weight.y, weight.z)), 0.95);
            if (rnd() > p)
                break;
            weight = weight / p;
        }
    }
    return color;
}

// ============================ 5. Scenes ============================
Vec3 camFrom, camAt;
double camFov, aspect, aperture = 0, focusDist = 1, exposure = 1;
bool sky;

void quad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Material m)
{
    objects.push_back(triangle(a, b, c, m));
    objects.push_back(triangle(a, c, d, m));
}
void areaLight(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 color)
{ // lamp: visible geometry + a sampling record
    quad(a, b, c, d, Material(LIGHT, color));
    Light L;
    L.corner = a;
    L.e1 = b - a;
    L.e2 = d - a;
    L.color = color;
    Vec3 nrm = cross(L.e1, L.e2);
    L.area = length(nrm);
    L.normal = nrm / L.area;
    lights.push_back(L);
}

void buildScene(int id)
{
    const Material glass(GLASS, Vec3(1, 1, 1), 1.5);
    if (id == 1)
    { // random spheres lit by the sky
        camFrom = Vec3(13, 2, 3);
        camAt = Vec3(0, 0.5, 0);
        camFov = 25;
        aspect = 16.0 / 9;
        sky = true;
        aperture = 0.08;
        focusDist = length(camFrom - Vec3(4, 1, 0));
        exposure = 1.3;
        objects.push_back(sphere(Vec3(0, -1000, 0), 1000, Material(DIFFUSE, Vec3(0.5, 0.5, 0.5))));
        rng.seed(42);
        for (int a = -8; a <= 8; a++)
            for (int b = -8; b <= 8; b++)
            {
                Vec3 c(a + 0.9 * rnd(), 0.2, b + 0.9 * rnd());
                double d1 = hypot(c.x, c.z), d2 = hypot(c.x + 4, c.z), d3 = hypot(c.x - 4, c.z);
                if (min(d1, min(d2, d3)) < 1.3)
                    continue;
                double p = rnd();
                Material m = glass;
                if (p < 0.65)
                {
                    double r = rnd() * rnd(), g = rnd() * rnd(), b2 = rnd() * rnd();
                    m = Material(DIFFUSE, Vec3(r, g, b2));
                }
                else if (p < 0.85)
                {
                    double r = rnd(0.5, 1), g = rnd(0.5, 1), b2 = rnd(0.5, 1);
                    m = Material(METAL, Vec3(r, g, b2), rnd(0, 0.4));
                }
                objects.push_back(sphere(c, 0.2, m));
            }
        objects.push_back(sphere(Vec3(0, 1, 0), 1, glass));
        objects.push_back(sphere(Vec3(-4, 1, 0), 1, Material(DIFFUSE, Vec3(0.4, 0.2, 0.1))));
        objects.push_back(sphere(Vec3(4, 1, 0), 1, Material(METAL, Vec3(0.7, 0.6, 0.5), 0)));
    }
    else
    { // Cornell box
        camFrom = Vec3(5, 5, -16);
        camAt = Vec3(5, 5, 5);
        camFov = 36;
        aspect = 1;
        sky = false;
        aperture = 0;
        exposure = 1.0;
        Material white(DIFFUSE, Vec3(0.73, 0.73, 0.73)), red(DIFFUSE, Vec3(0.65, 0.05, 0.05));
        Material green(DIFFUSE, Vec3(0.12, 0.45, 0.15));
        quad(Vec3(0, 0, 0), Vec3(10, 0, 0), Vec3(10, 0, 10), Vec3(0, 0, 10), white);
        quad(Vec3(0, 10, 0), Vec3(10, 10, 0), Vec3(10, 10, 10), Vec3(0, 10, 10), white);
        quad(Vec3(0, 0, 10), Vec3(10, 0, 10), Vec3(10, 10, 10), Vec3(0, 10, 10), white);
        quad(Vec3(0, 0, 0), Vec3(0, 0, 10), Vec3(0, 10, 10), Vec3(0, 10, 0), green);
        quad(Vec3(10, 0, 0), Vec3(10, 0, 10), Vec3(10, 10, 10), Vec3(10, 10, 0), red);
        areaLight(Vec3(3.5, 9.98, 3.5), Vec3(6.5, 9.98, 3.5), Vec3(6.5, 9.98, 6.5), Vec3(3.5, 9.98, 6.5),
                  Vec3(15, 15, 15));
        objects.push_back(sphere(Vec3(3.2, 1.8, 3.5), 1.8, glass));
        objects.push_back(sphere(Vec3(7, 2.2, 6), 2.2, Material(METAL, Vec3(0.8, 0.8, 0.85), 0.05)));
        objects.push_back(sphere(Vec3(6.2, 0.8, 2), 0.8, Material(DIFFUSE, Vec3(0.2, 0.3, 0.8))));
    }
    for (size_t i = 0; i < objects.size(); i++)
        finalize(objects[i]);
}

// ============================ 6. Output (PNG, no zlib needed) ============================
uint32_t crcTable[256];
void initCrc()
{
    for (uint32_t n = 0; n < 256; n++)
    {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crcTable[n] = c;
    }
}
void put32(vector<unsigned char> &v, uint32_t x)
{
    v.push_back(x >> 24);
    v.push_back(x >> 16);
    v.push_back(x >> 8);
    v.push_back(x);
}
void pngChunk(ofstream &f, const char *type, const vector<unsigned char> &data)
{
    vector<unsigned char> buf(type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < buf.size(); i++)
        c = crcTable[(c ^ buf[i]) & 255] ^ (c >> 8);
    vector<unsigned char> out;
    put32(out, (uint32_t)data.size());
    out.insert(out.end(), buf.begin(), buf.end());
    put32(out, c ^ 0xffffffffu);
    f.write((const char *)&out[0], out.size());
}
void savePNG(const char *file, int w, int h, const vector<unsigned char> &rgb)
{
    initCrc();
    vector<unsigned char> raw; // each scanline starts with filter byte 0
    for (int y = 0; y < h; y++)
    {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + y * w * 3, rgb.begin() + (y + 1) * w * 3);
    }
    vector<unsigned char> z; // zlib stream made of "stored" (uncompressed) deflate blocks
    z.push_back(0x78);
    z.push_back(0x01);
    for (size_t pos = 0; pos < raw.size() || pos == 0;)
    {
        size_t len = min<size_t>(65535, raw.size() - pos);
        bool last = pos + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(len & 255);
        z.push_back(len >> 8);
        z.push_back(~len & 255);
        z.push_back((~len >> 8) & 255);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
        if (last)
            break;
    }
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw.size(); i++)
    {
        a = (a + raw[i]) % 65521;
        b = (b + a) % 65521;
    }
    put32(z, (b << 16) | a);

    ofstream f(file, ios::binary);
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write((const char *)sig, 8);
    vector<unsigned char> ihdr;
    put32(ihdr, w);
    put32(ihdr, h);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0); // 8-bit RGB
    pngChunk(f, "IHDR", ihdr);
    pngChunk(f, "IDAT", z);
    pngChunk(f, "IEND", vector<unsigned char>());
}

unsigned char toByte(double v)
{ // ACES filmic tone map, then sRGB encode
    v *= exposure;
    v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
    v = min(max(v, 0.0), 1.0);
    v = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055;
    return (unsigned char)(v * 255 + 0.5);
}

// ============================ 7. Main ============================
int main(int argc, char **argv)
{
    int sceneId = argc > 1 ? atoi(argv[1]) : 1;
    int width = argc > 2 ? atoi(argv[2]) : 800;
    int maxSpp = argc > 3 ? atoi(argv[3]) : 200;
    int numThreads = argc > 4 ? atoi(argv[4]) : (int)thread::hardware_concurrency();
    useBVH = argc > 5 ? atoi(argv[5]) != 0 : true;
    double noise = argc > 6 ? atof(argv[6]) : 0.03;
    numThreads = max(numThreads, 1);
    maxSpp = max(maxSpp, 1);
    width = max(width, 16);
    const int MIN_SPP = min(16, maxSpp), BATCH = 16, TILE = 16;

    buildScene(sceneId);
    int height = max(1, int(width / aspect));
    cout << objects.size() << " objects, " << width << "x" << height << ", up to " << maxSpp << " spp, "
         << numThreads << " threads, BVH " << (useBVH ? "on" : "off") << endl;

    if (useBVH)
    {
        auto t0 = chrono::steady_clock::now();
        build(0, objects.size());
        cout << "BVH: " << nodes.size() << " nodes, built in " << fixed << setprecision(1)
             << chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count() << " ms" << endl;
    }

    // Camera (thin lens: aperture > 0 gives depth of field)
    Vec3 w = unit(camFrom - camAt), u = unit(cross(Vec3(0, 1, 0), w)), v = cross(w, u);
    double viewH = 2 * tan(camFov * 3.14159265358979 / 360), viewW = aspect * viewH;

    vector<Vec3> image(width * height);
    int tilesX = (width + TILE - 1) / TILE, tilesY = (height + TILE - 1) / TILE, numTiles = tilesX * tilesY;
    atomic<int> nextTile(0), doneTiles(0);
    atomic<long long> totalRays(0), totalSamples(0);

    auto work = [&]()
    {
        long long samples = 0;
        for (int t; (t = nextTile++) < numTiles;)
        {
            rng.seed(t + 1); // per-tile seed: identical image for any thread count
            int x0 = (t % tilesX) * TILE, y0 = (t / tilesX) * TILE;
            for (int y = y0; y < min(y0 + TILE, height); y++)
                for (int x = x0; x < min(x0 + TILE, width); x++)
                {
                    Vec3 sum;
                    double sumL = 0, sumL2 = 0;
                    int n = 0;
                    while (n < maxSpp)
                    {
                        int batch = min(BATCH, maxSpp - n);
                        for (int s = 0; s < batch; s++)
                        {
                            double px = (x + rnd()) / width - 0.5, py = 0.5 - (y + rnd()) / height;
                            Vec3 target = camFrom + focusDist * (px * viewW * u + py * viewH * v - w);
                            Vec3 lens = 0.5 * aperture * randomInDisk();
                            Ray ray;
                            ray.o = camFrom + lens.x * u + lens.y * v;
                            ray.d = unit(target - ray.o);
                            Vec3 c = trace(ray, sky);
                            double peak = max(c.x, max(c.y, c.z));
                            if (peak > 100)
                                c = (100 / peak) * c; // tame rare fireflies
                            sum = sum + c;
                            double l = luminance(c);
                            sumL += l;
                            sumL2 += l * l;
                        }
                        n += batch;
                        if (n >= MIN_SPP && n < maxSpp)
                        { // is this pixel already converged?
                            double mean = sumL / n;
                            double var = max(0.0, (sumL2 / n - mean * mean) * n / (n - 1));
                            if (sqrt(var / n) < noise * (0.05 + mean))
                                break;
                        }
                    }
                    image[y * width + x] = sum / n;
                    samples += n;
                }
            doneTiles++;
        }
        totalRays += rayCount;
        totalSamples += samples;
    };

    auto start = chrono::steady_clock::now();
    vector<thread> pool;
    for (int i = 0; i < numThreads; i++)
        pool.push_back(thread(work));
    while (doneTiles < numTiles)
    { // progress bar
        this_thread::sleep_for(chrono::milliseconds(150));
        double frac = double(doneTiles) / numTiles;
        double el = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        int bar = int(frac * 30);
        cerr << "\r[" << string(bar, '#') << string(30 - bar, '.') << "] " << int(frac * 100) << "%  ETA "
             << (frac > 0.02 ? int(el / frac - el) : -1) << " s   " << flush;
    }
    for (size_t i = 0; i < pool.size(); i++)
        pool[i].join();
    cerr << "\r" << string(60, ' ') << "\r";
    double secs = chrono::duration<double>(chrono::steady_clock::now() - start).count();

    cout << "Render time: " << fixed << setprecision(2) << secs << " s" << endl;
    cout << "Average samples/pixel: " << setprecision(1) << double(totalSamples) / (double(width) * height)
         << " (budget " << maxSpp << ")" << endl;
    cout << "Rays traced: " << setprecision(1) << totalRays / 1e6 << " M  ("
         << totalRays / 1e6 / secs << " Mrays/s)" << endl;

    vector<unsigned char> rgb(width * height * 3);
    for (int i = 0; i < width * height; i++)
    {
        rgb[i * 3] = toByte(image[i].x);
        rgb[i * 3 + 1] = toByte(image[i].y);
        rgb[i * 3 + 2] = toByte(image[i].z);
    }
    savePNG("output.png", width, height, rgb);
    cout << "Saved output.png" << endl;
    return 0;
}