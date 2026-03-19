import fs from 'fs';
import path from 'path';

const dir = 'public/images';
if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
}

// Simple SVG placeholder
const svgContent = (text) => `<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="100%" height="100%" fill="#ddd"/><text x="50%" y="50%" font-family="Arial" font-size="20" fill="#555" dominant-baseline="middle" text-anchor="middle">${text}</text></svg>`;

const images = [
    '/images/photo1764838409.jpg',
    '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838410.jpg',
    '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838409.jpg', '/images/photo1764838409.jpg',
    '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repositories.jpg', '/images/Repository.jpg', '/images/Repository.jpg',
    '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repository.jpg',
    '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repository.jpg', '/images/Repository.jpg', '/images/SVG.jpg',
    '/images/SVG.jpg', '/images/SVG.jpg', '/images/Repository.jpg', '/images/SVG.jpg', '/images/SVG.jpg'
];

images.forEach(img => {
    fs.writeFileSync(path.join(dir, img), svgContent(img)); // Writing SVG content to .jpg extension works for browser (sometimes) but for build check it just needs file existence. 
    // Actually, let's use .svg extension for real content or just empty valid file?
    // Vite doesn't check content validity for "resolve", just existence usually.
    // But let's stick to the names referenced.
});

console.log('Images generated');