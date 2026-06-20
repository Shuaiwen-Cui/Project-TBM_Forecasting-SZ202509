// TBM特征配置 - 包含特征名称、传输名和单位信息
const FEATURE_CONFIG = [
    { id: 1, name: '贯入度', transmission: 'date120', unit: 'mm/rpm', status: 'found' },
    { id: 2, name: '推进区间的压力（上）', transmission: 'date16', unit: 'MPa', status: 'found' },
    { id: 3, name: '推进区间的压力（右）', transmission: 'date17', unit: 'MPa', status: 'found' },
    { id: 4, name: '推进区间的压力（下）', transmission: 'date18', unit: 'MPa', status: 'found' },
    { id: 5, name: '推进区间的压力（左）', transmission: 'date19', unit: 'MPa', status: 'found' },
    { id: 6, name: '土舱土压（右）', transmission: 'date29', unit: 'MPa', status: 'found' },
    { id: 7, name: '土舱土压（右下）', transmission: 'date30', unit: 'MPa', status: 'found' },
    { id: 8, name: '土舱土压（左）', transmission: 'date31', unit: 'MPa', status: 'found' },
    { id: 9, name: '土舱土压（左下）', transmission: 'date32', unit: 'MPa', status: 'found' },
    { id: 10, name: 'No.16推进千斤顶速度', transmission: 'date7', unit: 'mm/min', status: 'found' },
    { id: 11, name: 'No.4推进千斤顶速度', transmission: 'date8', unit: 'mm/min', status: 'found' },
    { id: 12, name: 'No.8推进千斤顶速度', transmission: 'date9', unit: 'mm/min', status: 'found' },
    { id: 13, name: 'No.12推进千斤顶速度', transmission: 'date10', unit: 'mm/min', status: 'found' },
    { id: 14, name: '推进油缸总推力', transmission: 'date12', unit: 'kN', status: 'found' },
    { id: 15, name: 'No.16推进千斤顶行程', transmission: 'date3', unit: 'mm', status: 'found' },
    { id: 16, name: 'No.4推进千斤顶行程', transmission: 'date4', unit: 'mm', status: 'found' },
    { id: 17, name: 'No.8推进千斤顶行程', transmission: 'date5', unit: 'mm', status: 'found' },
    { id: 18, name: 'No.12推进千斤顶行程', transmission: 'date6', unit: 'mm', status: 'found' },
    { id: 19, name: '推进平均速度', transmission: 'date78', unit: 'mm/min', status: 'found' },
    { id: 20, name: '刀盘转速', transmission: 'date76', unit: 'r/min', status: 'found' },
    { id: 21, name: '刀盘扭矩', transmission: 'date77', unit: 'kN·m', status: 'found' },
    { id: 22, name: 'No.1刀盘电机扭矩', transmission: 'date47', unit: '%', status: 'found' },
    { id: 23, name: 'No.2刀盘电机扭矩', transmission: 'date48', unit: '%', status: 'found' },
    { id: 24, name: 'No.3刀盘电机扭矩', transmission: 'date49', unit: '%', status: 'found' },
    { id: 25, name: 'No.4刀盘电机扭矩', transmission: 'date50', unit: '%', status: 'found' },
    { id: 26, name: 'No.5刀盘电机扭矩', transmission: 'date51', unit: '%', status: 'found' },
    { id: 27, name: 'No.6刀盘电机扭矩', transmission: 'date52', unit: '%', status: 'found' },
    { id: 28, name: 'No.7刀盘电机扭矩', transmission: 'date53', unit: '%', status: 'found' },
    { id: 29, name: 'No.8刀盘电机扭矩', transmission: 'date54', unit: '%', status: 'found' },
    { id: 30, name: 'No.9刀盘电机扭矩', transmission: 'date55', unit: '%', status: 'found' },
    { id: 31, name: 'No.10刀盘电机扭矩', transmission: 'date56', unit: '%', status: 'found' }
];

// 特征分类配置
const FEATURE_CATEGORIES = {
    '贯入度': [1],
    '推进压力': [2, 3, 4, 5],
    '土舱土压': [6, 7, 8, 9],
    '推进千斤顶速度': [10, 11, 12, 13],
    '推进千斤顶行程': [15, 16, 17, 18],
    '推进系统': [14, 19],
    '刀盘系统': [20, 21],
    '刀盘电机扭矩': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
};

// 数据状态配置
const DATA_STATUS = {
    'valid': { text: '正常', class: 'status-valid', icon: 'fas fa-check-circle' },
    'missing': { text: '缺失', class: 'status-missing', icon: 'fas fa-exclamation-triangle' },
    'predicted': { text: '预测', class: 'status-predicted', icon: 'fas fa-brain' },
    'error': { text: '错误', class: 'status-error', icon: 'fas fa-times-circle' }
};

// 趋势配置
const TREND_CONFIG = {
    'up': { icon: 'fas fa-arrow-up', class: 'trend-up' },
    'down': { icon: 'fas fa-arrow-down', class: 'trend-down' },
    'stable': { icon: 'fas fa-minus', class: 'trend-stable' },
    'unknown': { icon: 'fas fa-question', class: 'trend-unknown' }
};
